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

#define HARRIS_THRESHOLD 15.0f
// TODO: is there a way to define these in one file?
#define BRIEFN 512
// TODO: Not sure what the optimal value here is, neither the BRIEF nor the ORB
// paper mentions one (although the ORB paper data suggests 64).
#define DISTANCE_THRESHOLD 90

typedef struct PointPair {
    int2 p1;
    int2 p2;
} PointPair;

typedef struct Vector {
    PointPair p;
    float magnitude;
    // Angle is in degrees
    float angle;
} Vector;

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE |
                          CLK_FILTER_NEAREST;

// Returns the averaged luminance (grayscale) value at the given point.
float luminance(image2d_t src, int2 loc) {
    float4 pixel = read_imagef(src, sampler, loc);
    return (pixel.x + pixel.y + pixel.z) / 3.0f;
}

float convolve(image2d_t src, int2 loc, int mask[3][3]) {
    float ret = 0;

    // These loops touch each pixel surrounding loc as well as loc itself
    for (int i = 1, i2 = 0; i >= -1; --i, ++i2) {
        for (int j = -1, j2 = 0; j <= 1; ++j, ++j2) {
            ret += mask[i2][j2] * luminance(src, (int2)(loc.x + j, loc.y + i));
        }
    }

    return ret;
}

// Sums dx * dy for all pixels within radius of loc
float sum_deriv_prod(image2d_t src, int2 loc, int mask_x[3][3], int mask_y[3][3], int radius) {
    float ret = 0;

    for (int i = radius; i >= -radius; --i) {
        for (int j = -radius; j <= radius; ++j) {
            ret += convolve(src, (int2)(loc.x + j, loc.y + i), mask_x) *
                   convolve(src, (int2)(loc.x + j, loc.y + i), mask_y);
        }
    }

    return ret;
}

// Sums d<>^2 (determined by mask) for all pixels within radius of loc
float sum_deriv_pow(image2d_t src, int2 loc, int mask[3][3], int radius) {
    float ret = 0;

    for (int i = radius; i >= -radius; --i) {
        for (int j = -radius; j <= radius; ++j) {
            ret += pow(
                convolve(src, (int2)(loc.x + j, loc.y + i), mask),
                2
            );
        }
    }

    return ret;
}

// Fills a box with the given radius and pixel around loc
void draw_box(__write_only image2d_t dst, int2 loc, float4 pixel, int radius) {
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            write_imagef(
                dst,
                (int2)(
                    // Clamp to avoid writing outside image bounds
                    clamp(loc.x + i, 0, (int)get_global_size(0) - 1),
                    clamp(loc.y + j, 0, (int)get_global_size(1) - 1)
                ),
                pixel
            );
        }
    }
}

// Use this when the global work size does not match the dimensions of dst
void draw_box_plus(__write_only image2d_t dst, int2 loc, float4 pixel, int radius, int size_x, int size_y) {
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            write_imagef(
                dst,
                (int2)(
                    // Clamp to avoid writing outside image bounds
                    clamp(loc.x + i, 0, size_x - 1),
                    clamp(loc.y + j, 0, size_y - 1)
                ),
                pixel
            );
        }
    }
}


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

// This kernel computes the harris response for the given src image within the
// given radius and writes it to harris_buf
// TODO: src and dst are just for debugging, remove or improve when finished
__kernel void harris_response(
    __read_only  image2d_t src,
    __write_only image2d_t dst,
    __global float *harris_buf,
    int radius
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float scale = 1.0f / ((1 << 2) * radius * 255.0f);

    int sobel_mask_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobel_mask_y[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    float sumdxdy = sum_deriv_prod(src, loc, sobel_mask_x, sobel_mask_y, radius);
    float sumdx2 = sum_deriv_pow(src, loc, sobel_mask_x, radius);
    float sumdy2 = sum_deriv_pow(src, loc, sobel_mask_y, radius);

    float trace = sumdx2 + sumdy2;
    // r = det(M) - k(trace(M))^2
    // k usually between 0.04 to 0.06
    // threshold around 5?
    float r = (sumdx2 * sumdy2 - pow(sumdxdy, 2)) - 0.04f * (trace * trace) * pow(scale, 4);

    // This is the shi-tomasi method of calculating r
    // threshold around 2.5?
    // float r = min(sumdx2, sumdy2);

    // Threshold the r value
    if (r > HARRIS_THRESHOLD) {
        // draw_box(dst, loc, (float4)(0.0f, 1.0f, 0.0f, 1.0f), 5);
        write_to_1d_arrf(harris_buf, loc, r);
    } else {
        write_to_1d_arrf(harris_buf, loc, 0.0f);
    }

    // float4 pixel = read_imagef(src, sampler, loc);
    // write_imagef(dst, loc, pixel);
}

// Performs non-maximum suppression on the given buffer (buffer is expected to
// represent harris response for an image) and writes the output to another.
//
// This means that for each response value, it checks all
// surrounding values within a hardcoded window and rejects the value if it is
// not the largest within that window.
// TODO: src and dst are just for debugging, remove or improve when finished
__kernel void nonmax_suppression(
    __read_only  image2d_t src,
    __write_only image2d_t dst,
    __global const float *harris_buf,
    __global float *harris_buf_suppressed
) {
    const int window_size = 30;
    const int half_window = window_size / 2;

    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float center_val = read_from_1d_arrf(harris_buf, loc);

    if (center_val == 0.0f) {
        // obviously not a maximum
        write_to_1d_arrf(harris_buf_suppressed, loc, center_val);
        goto done;
    }

    // TODO: could save an iteration by not comparing the center value to itself
    for (int i = -half_window; i <= half_window; ++i) {
        for (int j = -half_window; j <= half_window; ++j) {
            if (center_val < read_from_1d_arrf(harris_buf, (int2)(loc.x + i, loc.y + j))) {
                // This value is not the maximum within the window
                write_to_1d_arrf(harris_buf_suppressed, loc, 0.0f);
                goto done;
            }
        }
    }

    write_to_1d_arrf(harris_buf_suppressed, loc, center_val);
    goto done;

done: // debug stuff
    // write_imagef(dst, loc, center_val, loc));
    
    if (center_val != 0.0f) {
        // draw_box(dst, loc, (float4)(1.0f, 0.0f, 0.0f, 1.0f), 5);
    }
    float4 pixel = read_imagef(src, sampler, loc);
    write_imagef(dst, loc, pixel);
}

// Extracts BRIEF descriptors from the src image for the given features using the
// provided sampler.
__kernel void brief_descriptors(
    __read_only image2d_t src,
    // TODO: debug only
    __write_only image2d_t dst,
    __global const float *features,
    // TODO: changing BRIEFN will make this a different type, figure out how to
    // deal with that
    __global ulong8 *desc_buf,
    __global const PointPair *brief_pattern
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    // TODO: restructure data so we don't have to do this
    if (read_from_1d_arrf(features, loc) == 0.0f) {
        write_to_1d_arrul8(desc_buf, loc, (ulong8)(0));
        return;
    }

    // TODO: this code is hardcoded for ulong8
    ulong8 desc = 0;
    ulong *p = &desc;

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 64; ++j) {
            PointPair pair = brief_pattern[j * (i + 1)];
            float l1 = luminance(src, (int2)(loc.x + pair.p1.x, loc.y + pair.p1.y));
            float l2 = luminance(src, (int2)(loc.x + pair.p2.x, loc.y + pair.p2.y));

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
    __read_only image2d_t src,
    __write_only image2d_t dst,
    __global const ulong8 *desc_buf,
    __global const ulong8 *prev_desc_buf,
    __global Vector *matches_buf,
    int search_radius
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    ulong8 desc = read_from_1d_arrul8(desc_buf, loc);
    Vector invalid_vector = (Vector) {
        (PointPair) {
            (int2)(-1, -1),
            (int2)(-1, -1)
        },
        -1,
        -1
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
    
    bool has_compared = false;

    // Cyan box: Feature point in current frame
    // draw_box(dst, loc, (float4)(0.0f, 0.5f, 0.5f, 1.0f), 5);

    // TODO: this could potentially search in a more optimal way
    for (int i = -search_radius; i < search_radius; ++i) {
        for (int j = -search_radius; j < search_radius; ++j) {
            int2 prev_point = (int2)(loc.x + i, loc.y + j);
            int total_dist = 0;

            ulong8 prev_desc = read_from_1d_arrul8(prev_desc_buf, prev_point);

            if (prev_desc.s0 == 0 && prev_desc.s1 == 0) {
                continue;
            }

            // Orange box: potential match point from previous frame
            // draw_box(dst, prev_point, (float4)(0.7f, 0.3f, 0.0f, 1.0f), 3);
            has_compared = true;

            total_dist += popcount(desc.s0 ^ prev_desc.s0);
            total_dist += popcount(desc.s1 ^ prev_desc.s1);
            total_dist += popcount(desc.s2 ^ prev_desc.s2);
            total_dist += popcount(desc.s3 ^ prev_desc.s3);
            total_dist += popcount(desc.s4 ^ prev_desc.s4);
            total_dist += popcount(desc.s5 ^ prev_desc.s5);
            total_dist += popcount(desc.s6 ^ prev_desc.s6);
            total_dist += popcount(desc.s7 ^ prev_desc.s7);

            if (total_dist < DISTANCE_THRESHOLD) {
                float dx = (float)(loc.x - prev_point.x);
                float dy = (float)(loc.y - prev_point.y);
                float angle = (atan2(dy, dx) * 180.0f) / M_PI_F;

                write_to_1d_arrvec(
                    matches_buf,
                    loc,
                    (Vector) {
                        (PointPair) {
                            prev_point,
                            loc
                        },
                        sqrt(dx * dx + dy * dy),
                        angle
                    }
                );

                // Debug stuff
                // Green box: point that was matched to a point in the previous frame
                // draw_box(dst, loc, (float4)(0.0f, 1.0f, 0.0f, 1.0f), 5);
                // Blue box: said point in previous frame
                // draw_box(dst, prev_point, (float4)(0.0f, 0.0f, 1.0f, 1.0f), 3);

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

    if (has_compared == false) {
        // Red box: point that has nothing to compare to in the previous frame
        // draw_box(dst, loc, (float4)(1.0f, 0.0f, 0.0f, 1.0f), 5);
    }
}

// For debugging. Draws boxes to display point matches
__kernel void debug_matches(
    __write_only image2d_t dst,
    int image_size_x,
    int image_size_y,
    __global const Vector *matches_contig
) {
    size_t idx = get_global_id(0);
    Vector v = matches_contig[idx];

    if (v.angle == -1) {
        // Orange box: point that was matched to a point in the previous frame but is invalid because of angle
        draw_box_plus(dst, v.p.p2, (float4)(1.0f, 0.5f, 0.0f, 1.0f), 3, image_size_x, image_size_y);
        // Light blue box: said point in previous frame
        draw_box_plus(dst, v.p.p1, (float4)(0.0f, 0.3f, 0.7f, 1.0f), 1, image_size_x, image_size_y);

        return;
    }


    if (v.magnitude == -1) {
        // Purple box: point that was matched to a point in the previous frame but is invalid because of magnitude
        draw_box_plus(dst, v.p.p2, (float4)(0.5f, 0.0f, 1.0f, 1.0f), 3, image_size_x, image_size_y);
        // Light blue box: said point in previous frame
        draw_box_plus(dst, v.p.p1, (float4)(0.0f, 0.3f, 0.7f, 1.0f), 1, image_size_x, image_size_y);

        return;
    }

    // Green box: point that was matched to a point in the previous frame
    draw_box_plus(dst, v.p.p2, (float4)(0.0f, 1.0f, 0.0f, 1.0f), 5, image_size_x, image_size_y);
    // Blue box: said point in previous frame
    draw_box_plus(dst, v.p.p1, (float4)(0.0f, 0.0f, 1.0f, 1.0f), 3, image_size_x, image_size_y);
}
