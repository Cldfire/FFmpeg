/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#define HARRIS_THRESHOLD 3.0f
// TODO: is there a way to define these in one file?
#define BRIEFN 512
// TODO: Not sure what the optimal value here is, neither the BRIEF nor the ORB
// paper mentions one (although the ORB paper data suggests 64).
#define DISTANCE_THRESHOLD 80

// Sub-pixel refinement window for feature points
#define REFINE_WIN_HALF_W 5
#define REFINE_WIN_HALF_H 5
#define REFINE_WIN_W 11 // REFINE_WIN_HALF_W * 2 + 1
#define REFINE_WIN_H 11

// Non-maximum suppression window size
#define NONMAX_WIN 30
#define NONMAX_WIN_HALF 15 // NONMAX_WIN / 2

typedef struct PointPair {
    // Previous frame
    float2 p1;
    // Current frame
    float2 p2;
} PointPair;

typedef struct SmoothedPointPair {
    // Non-smoothed point in current frame
    int2 p1;
    // Smoothed point in current frame
    float2 p2;
} SmoothedPointPair;

typedef struct Vector {
    PointPair p;
    // Used to mark vectors as potential outliers
    int should_consider;
} Vector;

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE |
                          CLK_FILTER_NEAREST;

const sampler_t sampler_linear = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE |
                          CLK_FILTER_LINEAR;

const sampler_t sampler_linear_mirror = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_MIRRORED_REPEAT |
                          CLK_FILTER_LINEAR;

// Writes to a 1D array at loc, treating it as a 2D array with the same
// dimensions as the global work size.
void write_to_1d_arrf(__global float *buf, int2 loc, float val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

void write_to_1d_arrul8(__global ulong8 *buf, int2 loc, ulong8 val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

void write_to_1d_arrvec(__global Vector *buf, int2 loc, Vector val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

void write_to_1d_arrf2(__global float2 *buf, int2 loc, float2 val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

// Above except reading
float read_from_1d_arrf(__global const float *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

ulong8 read_from_1d_arrul8(__global const ulong8 *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

Vector read_from_1d_arrvec(__global const Vector *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

float2 read_from_1d_arrf2(__global const float2 *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

// Returns the grayscale value at the given point.
float pixel_grayscale(__read_only image2d_t src, int2 loc) {
    float4 pixel = read_imagef(src, sampler, loc);
    return (pixel.x + pixel.y + pixel.z) / 3.0f;
}

float convolve(__read_only image2d_t grayscale, int2 loc, float mask[3][3])
{
    float ret = 0;

    int start_x = loc.x - 1;
    int end_x   = loc.x + 1;
    int start_y = loc.y + 1;
    int end_y   = loc.y - 1;

    // These loops touch each pixel surrounding loc as well as loc itself
    for (int i = start_y, i2 = 0; i >= end_y; --i, ++i2) {
        for (int j = start_x, j2 = 0; j <= end_x; ++j, ++j2) {
            ret += mask[i2][j2] * read_imagef(grayscale, sampler, (int2)(j, i)).x;
        }
    }

    return ret;
}

// Sums dx * dy for all pixels within radius of loc
float sum_deriv_prod(__read_only image2d_t grayscale, int2 loc, float mask_x[3][3], float mask_y[3][3], int radius)
{
    float ret = 0;

    for (int i = radius; i >= -radius; --i) {
        for (int j = -radius; j <= radius; ++j) {
            ret += convolve(grayscale, (int2)(loc.x + j, loc.y + i), mask_x) *
                   convolve(grayscale, (int2)(loc.x + j, loc.y + i), mask_y);
        }
    }

    return ret;
}

// Sums d<>^2 (determined by mask) for all pixels within radius of loc
float sum_deriv_pow(__read_only image2d_t grayscale, int2 loc, float mask[3][3], int radius)
{
    float ret = 0;

    for (int i = radius; i >= -radius; --i) {
        for (int j = -radius; j <= radius; ++j) {
            float deriv = convolve(grayscale, (int2)(loc.x + j, loc.y + i), mask);
            ret += deriv * deriv;
        }
    }

    return ret;
}

// Fills a box with the given radius and pixel around loc
void draw_box(__write_only image2d_t dst, int2 loc, float4 pixel, int radius)
{
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            write_imagef(
                dst,
                (int2)(
                    // Clamp to avoid writing outside image bounds
                    clamp(loc.x + i, 0, get_image_dim(dst).x - 1),
                    clamp(loc.y + j, 0, get_image_dim(dst).y - 1)
                ),
                pixel
            );
        }
    }
}

// Converts the src image to grayscale
__kernel void grayscale(
    __read_only image2d_t src,
    __write_only image2d_t grayscale
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    write_imagef(grayscale, loc, (float4)(pixel_grayscale(src, loc), 0.0f, 0.0f, 1.0f));
}

// This kernel computes the harris response for the given grayscale src image
// within the given radius and writes it to harris_buf
__kernel void harris_response(
    __read_only image2d_t grayscale,
    __global float *harris_buf,
    int radius
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float scale = 1.0f / ((1 << 2) * radius * 255.0f);

    float sobel_mask_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    float sobel_mask_y[3][3] = {
        { 1,   2,  1},
        { 0,   0,  0},
        {-1,  -2, -1}
    };

    float sumdxdy = sum_deriv_prod(grayscale, loc, sobel_mask_x, sobel_mask_y, radius);
    float sumdx2 = sum_deriv_pow(grayscale, loc, sobel_mask_x, radius);
    float sumdy2 = sum_deriv_pow(grayscale, loc, sobel_mask_y, radius);

    float trace = sumdx2 + sumdy2;
    // r = det(M) - k(trace(M))^2
    // k usually between 0.04 to 0.06
    float r = (sumdx2 * sumdy2 - sumdxdy * sumdxdy) - 0.04f * (trace * trace) * pow(scale, 4);

    // Threshold the r value
    write_to_1d_arrf(harris_buf, loc, r * step(HARRIS_THRESHOLD, r));
}

// Gets a patch centered around a float coordinate from a grayscale image using
// bilinear interpolation
void get_rect_sub_pix(
    __read_only image2d_t grayscale,
    float *buffer,
    int size_x,
    int size_y,
    float2 center
) {
    for (int i = 0; i < size_y; i++) {
        for (int j = 0; j < size_x; j++) {
            buffer[i * size_x + j] = read_imagef(
                grayscale,
                sampler_linear,
                (float2)(j + center.x - (size_x - 1) * 0.5, i + center.y - (size_y - 1) * 0.5)
            ).x * 255.0;
        }
    }
}

// Refines detected features at a sub-pixel level
//
// This function is ported from OpenCV
float2 corner_sub_pix(
    __read_only image2d_t grayscale,
    float *mask
) {
    // This is the location of the feature point we are refining
    float2 cI = (float2)(get_global_id(0), get_global_id(1));
    float2 cT = cI;
    int src_width = get_global_size(0);
    int src_height = get_global_size(1);

    const int max_iters = 40;
    const float eps = 0.001 * 0.001;
    int i, j, k;

    int iter = 0;
    float err = 0;
    float subpix[(REFINE_WIN_W + 2) * (REFINE_WIN_H + 2)];
    const float flt_epsilon = 0x1.0p-23f;

    do {
        float2 cI2;
        float a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;

        get_rect_sub_pix(grayscale, subpix, REFINE_WIN_W + 2, REFINE_WIN_H + 2, cI);
        const float *subpix_ptr = &subpix;
        subpix_ptr += REFINE_WIN_W + 2 + 1;

        // process gradient
        for (i = 0, k = 0; i < REFINE_WIN_H; i++, subpix_ptr += REFINE_WIN_W + 2) {
            float py = i - REFINE_WIN_HALF_H;

            for (j = 0; j < REFINE_WIN_W; j++, k++) {
                float m = mask[k];
                float tgx = subpix_ptr[j + 1] - subpix_ptr[j - 1];
                float tgy = subpix_ptr[j + REFINE_WIN_W + 2] - subpix_ptr[j - REFINE_WIN_W - 2];
                float gxx = tgx * tgx * m;
                float gxy = tgx * tgy * m;
                float gyy = tgy * tgy * m;
                float px = j - REFINE_WIN_HALF_W;

                a += gxx;
                b += gxy;
                c += gyy;

                bb1 += gxx * px + gxy * py;
                bb2 += gxy * px + gyy * py;
            }
        }

        float det = a * c - b * b;
        if (fabs(det) <= flt_epsilon * flt_epsilon) {
            break;
        }

        // 2x2 matrix inversion
        float scale = 1.0 / det;
        cI2.x = (float)(cI.x + (c * scale * bb1) - (b * scale * bb2));
        cI2.y = (float)(cI.y - (b * scale * bb1) + (a * scale * bb2));
        err = (cI2.x - cI.x) * (cI2.x - cI.x) + (cI2.y - cI.y) * (cI2.y - cI.y);

        cI = cI2;
        if (cI.x < 0 || cI.x >= src_width || cI.y < 0 || cI.y >= src_height) {
            break;
        }
    } while (++iter < max_iters && err > eps);

    // Make sure new point isn't too far from the initial point (indicates poor convergence)
    if (fabs(cI.x - cT.x) > REFINE_WIN_HALF_W || fabs(cI.y - cT.y) > REFINE_WIN_HALF_H) {
        cI = cT;
    }

    return cI;
}

// Performs non-maximum suppression on the harris response, refines the locations
// of the maximum values (strongest corners), and writes the resulting feature
// locations to refined_features.
__kernel void refine_features(
    __read_only image2d_t grayscale,
    __global const float *harris_buf,
    __global float2 *refined_features
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float center_val = read_from_1d_arrf(harris_buf, loc);

    if (center_val == 0.0f) {
        // obviously not a maximum
        write_to_1d_arrf2(refined_features, loc, (float2)(-1, -1));
        return;
    }

    int start_x = clamp(loc.x - NONMAX_WIN_HALF, 0, (int)get_global_size(0) - 1);
    int end_x   = clamp(loc.x + NONMAX_WIN_HALF, 0, (int)get_global_size(0) - 1);
    int start_y = clamp(loc.y - NONMAX_WIN_HALF, 0, (int)get_global_size(1) - 1);
    int end_y   = clamp(loc.y + NONMAX_WIN_HALF, 0, (int)get_global_size(1) - 1);

    // TODO: could save an iteration by not comparing the center value to itself
    for (int i = start_x; i <= end_x; ++i) {
        for (int j = start_y; j <= end_y; ++j) {
            if (center_val < read_from_1d_arrf(harris_buf, (int2)(i, j))) {
                // This value is not the maximum within the window
                write_to_1d_arrf2(refined_features, loc, (float2)(-1, -1));
                return;
            }
        }
    }

    // TODO: generate this once on the host
    float mask[REFINE_WIN_H * REFINE_WIN_W];
    for (int i = 0; i < REFINE_WIN_H; i++) {
        float y = (float)(i - REFINE_WIN_HALF_H) / REFINE_WIN_HALF_H;
        float vy = exp(-y * y);

        for (int j = 0; j < REFINE_WIN_W; j++) {
            float x = (float)(j - REFINE_WIN_HALF_W) / REFINE_WIN_HALF_W;
            mask[i * REFINE_WIN_W + j] = (float)(vy * exp(-x * x));
        }
    }

    float2 refined = corner_sub_pix(grayscale, mask);
    write_to_1d_arrf2(refined_features, loc, refined);
    // write_to_1d_arrf2(refined_features, loc, (float2)(loc.x, loc.y));
}

// Extracts BRIEF descriptors from the grayscale src image for the given features
// using the provided sampler.
__kernel void brief_descriptors(
    __read_only image2d_t grayscale,
    __global const float2 *refined_features,
    // TODO: changing BRIEFN will make this a different type, figure out how to
    // deal with that
    __global ulong8 *desc_buf,
    __global const PointPair *brief_pattern
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float2 feature = read_from_1d_arrf2(refined_features, loc);

    // TODO: restructure data so we don't have to do this
    if (feature.x == -1) {
        write_to_1d_arrul8(desc_buf, loc, (ulong8)(0));
        return;
    }

    // TODO: this code is hardcoded for ulong8
    ulong8 desc = 0;
    ulong *p = &desc;

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 64; ++j) {
            PointPair pair = brief_pattern[j * (i + 1)];
            float l1 = read_imagef(grayscale, sampler_linear, (float2)(feature.x + pair.p1.x, feature.y + pair.p1.y)).x;
            float l2 = read_imagef(grayscale, sampler_linear, (float2)(feature.x + pair.p2.x, feature.y + pair.p2.y)).x;

            if (l1 < l2) {
                p[i] |= 1UL << j;
            }
        }
    }

    write_to_1d_arrul8(desc_buf, loc, desc);
}

// Given buffers with descriptors for the current and previous frame, determines
// which ones match (looking in a box of search_radius size around each descriptor)
// and writes the resulting point correspondences to matches_buf.
// TODO: images are just for debugging, remove
__kernel void match_descriptors(
    __global const float2 *prev_refined_features,
    __global const float2 *refined_features,
    __global const ulong8 *desc_buf,
    __global const ulong8 *prev_desc_buf,
    __global Vector *matches_buf,
    int search_radius
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    ulong8 desc = read_from_1d_arrul8(desc_buf, loc);
    Vector invalid_vector = (Vector) {
        (PointPair) {
            (float2)(-1, -1),
            (float2)(-1, -1)
        },
        0
    };

    // TODO: restructure data so we don't have to do this
    // also this is an ugly hack
    if (desc.s0 == 0 && desc.s1 == 0) {
        write_to_1d_arrvec(
            matches_buf,
            loc,
            invalid_vector
        );
        return;
    }

    int start_x = clamp(loc.x - search_radius, 0, (int)get_global_size(0) - 1);
    int end_x = clamp(loc.x + search_radius, 0, (int)get_global_size(0) - 1);
    int start_y = clamp(loc.y - search_radius, 0, (int)get_global_size(1) - 1);
    int end_y = clamp(loc.y + search_radius, 0, (int)get_global_size(1) - 1);

    // TODO: this could potentially search in a more optimal way
    for (int i = start_x; i < end_x; ++i) {
        for (int j = start_y; j < end_y; ++j) {
            int2 prev_point = (int2)(i, j);
            int total_dist = 0;

            ulong8 prev_desc = read_from_1d_arrul8(prev_desc_buf, prev_point);

            if (prev_desc.s0 == 0 && prev_desc.s1 == 0) {
                continue;
            }

            total_dist += popcount(desc.s0 ^ prev_desc.s0);
            total_dist += popcount(desc.s1 ^ prev_desc.s1);
            total_dist += popcount(desc.s2 ^ prev_desc.s2);
            total_dist += popcount(desc.s3 ^ prev_desc.s3);
            total_dist += popcount(desc.s4 ^ prev_desc.s4);
            total_dist += popcount(desc.s5 ^ prev_desc.s5);
            total_dist += popcount(desc.s6 ^ prev_desc.s6);
            total_dist += popcount(desc.s7 ^ prev_desc.s7);

            if (total_dist < DISTANCE_THRESHOLD) {
                write_to_1d_arrvec(
                    matches_buf,
                    loc,
                    (Vector) {
                        (PointPair) {
                            read_from_1d_arrf2(prev_refined_features, prev_point),
                            read_from_1d_arrf2(refined_features, loc)
                        },
                        1
                    }
                );

                return;
            }
        }
    }

    // There is no found match for this point
    write_to_1d_arrvec(
        matches_buf,
        loc,
        invalid_vector
    );
}

// Returns the position of the given point after the transform is applied
float2 transformed_point(float2 p, __global const float *transform) {
    float2 ret;

    ret.x = p.x * transform[0] + p.y * transform[1] + transform[2];
    ret.y = p.x * transform[3] + p.y * transform[4] + transform[5];

    return ret;
}


// Performs the given transform on the src image
__kernel void transform(
    __read_only image2d_t src,
    __write_only image2d_t dst,
    __global const float *transform
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    write_imagef(
        dst,
        loc,
        read_imagef(
            src,
            sampler_linear_mirror,
            transformed_point((float2)(loc.x, loc.y), transform)
        )
    );
}

// Returns the new location of the given point using the given crop bounding box
// and the width and height of the original frame.
float2 cropped_point(
    float2 p,
    float2 top_left,
    float2 bottom_right,
    int2 orig_dim
) {
    float2 ret;

    float crop_width  = bottom_right.x - top_left.x;
    float crop_height = bottom_right.y - top_left.y;

    float width_percent = p.x / (float)orig_dim.x;
    float height_percent = p.y / (float)orig_dim.y;

    ret.x = (width_percent * crop_width) + top_left.x;
    ret.y = (height_percent * crop_height) + ((float)orig_dim.y - bottom_right.y);

    return ret;
}

// Upscales the given cropped region to the size of the original frame
// TODO: combine with transform to avoid interpolating twice?
__kernel void crop_upscale(
    __read_only image2d_t src,
    __write_only image2d_t dst,
    float2 top_left,
    float2 bottom_right
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    write_imagef(
        dst,
        loc,
        read_imagef(
            src,
            sampler_linear,
            cropped_point((float2)(loc.x, loc.y), top_left, bottom_right, get_image_dim(dst))
        )
    );
}

// Draws boxes to represent the given point matches and uses the given transform
// and crop info to make sure their positions are accurate on the transformed frame.
//
// model_matches is an array of three points that were used by the RANSAC process
// to generate the given transform
__kernel void draw_debug_info(
    __write_only image2d_t dst,
    __global const Vector *matches,
    __global const Vector *model_matches,
    int num_model_matches,
    float2 crop_top_left,
    float2 crop_bottom_right,
    __global const float *transform
) {
    int loc = get_global_id(0);
    Vector vec = matches[loc];
    // Black box: matched point that RANSAC considered an outlier
    float4 big_rect_color = (float4)(0.1f, 0.1f, 0.1f, 1.0f);

    if (vec.should_consider) {
        // Green box: matched point that RANSAC considered an inlier
        big_rect_color = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
    }

    for (int i = 0; i < num_model_matches; i++) {
        if (vec.p.p2.x == model_matches[i].p.p2.x && vec.p.p2.y == model_matches[i].p.p2.y) {
            // Orange box: point used to calculate model
            big_rect_color = (float4)(1.0f, 0.5f, 0.0f, 1.0f);
        }
    }

    float2 transformed_p1 = cropped_point(
        transformed_point(vec.p.p1, transform),
        crop_top_left,
        crop_bottom_right,
        get_image_dim(dst)
    );
    float2 transformed_p2 = cropped_point(
        transformed_point(vec.p.p2, transform),
        crop_top_left,
        crop_bottom_right,
        get_image_dim(dst)
    );

    draw_box(dst, (int2)(transformed_p2.x, transformed_p2.y), big_rect_color, 5);
    // Small light blue box: the point in the previous frame
    draw_box(dst, (int2)(transformed_p1.x, transformed_p1.y), (float4)(0.0f, 0.3f, 0.7f, 1.0f), 3);
}
