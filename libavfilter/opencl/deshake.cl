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

#define HARRIS_THRESHOLD 10.0f
// TODO: is there a way to define these in one file?
#define BRIEFN 512
// TODO: Not sure what the optimal value here is, neither the BRIEF nor the ORB
// paper mentions one (although the ORB paper data suggests 64).
#define DISTANCE_THRESHOLD 90

typedef struct DerivInfo {
    float dx;
    float dy;
    // sqrt(dx^2 + dy^2)
    float magnitude;
    // arctan(dy / dx)
    float gradient_direction;
} DerivInfo;

typedef struct PointPair {
    int2 p1;
    int2 p2;
} PointPair;

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

// Writes to a 1D array at loc, treating it as a 2D array with the same
// dimensions as the global work size.
void write_to_1d_arrf(__global float *buf, int2 loc, float val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

void write_to_1d_arrf2(__global float2 *buf, int2 loc, float2 val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

void write_to_1d_arrul8(__global ulong8 *buf, int2 loc, ulong8 val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

void write_to_1d_arrpointp(__global PointPair *buf, int2 loc, PointPair val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

void write_to_1d_arrdinfo(__global DerivInfo *buf, int2 loc, DerivInfo val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

// Above except reading
float read_from_1d_arrf(__global const float *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

float2 read_from_1d_arrf2(__global const float2 *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

ulong8 read_from_1d_arrul8(__global const ulong8 *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

DerivInfo read_from_1d_arrdinfo(__global const DerivInfo *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

// Sums dx * dy for all pixels surrounding loc
float sum_deriv_prod(__global const float2 *deriv_buf_suppressed, int2 loc) {
    float ret = 0;
    float2 d;

    // These loops touch each pixel surrounding loc as well as loc itself
    for (int i = 1; i >= -1; --i) {
        for (int j = -1; j <= 1; ++j) {
            d = read_from_1d_arrf2(deriv_buf_suppressed, (int2)(loc.x + j, loc.y + i));
            ret += d.x * d.y;
        }
    }

    return ret;
}

// Sums d<>^2 (pass true for dx, false for dy) for all pixels surrounding loc
float sum_deriv_pow(__global const float2 *deriv_buf_suppressed, int2 loc, bool dx) {
    float ret = 0;
    float d;

    // These loops touch each pixel surrounding loc as well as loc itself
    for (int i = 1; i >= -1; --i) {
        for (int j = -1; j <= 1; ++j) {
            // TODO: this if could be easily eliminated
            if (dx) {
                d = read_from_1d_arrf2(deriv_buf_suppressed, (int2)(loc.x + j, loc.y + i)).x;
            } else {
                d = read_from_1d_arrf2(deriv_buf_suppressed, (int2)(loc.x + j, loc.y + i)).y;
            }

            ret += d * d;
        }
    }

    return ret;
}

// This kernel computes the x and y derivatives along with the combined magnitude and
// gradient direction for each pixel of the src image and writes the data to deriv_buf
// TODO: dst just for debugging, remove or improve when finished
__kernel void derivative(
    __read_only  image2d_t src,
    __write_only image2d_t dst,
    __global DerivInfo *deriv_buf
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

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

    float dx = convolve(src, loc, sobel_mask_x);
    float dy = convolve(src, loc, sobel_mask_y);

    write_to_1d_arrdinfo(
        deriv_buf,
        loc,
        (DerivInfo) {
            dx,
            dy,
            sqrt(dx * dx + dy * dy),
            atan(dy / dx)
        }
    );
}

// Performs non-maximum suppression on the image derivative given deriv_buf, writing
// the resulting suppressed derivative to deriv_buf_suppressed
// TODO: src and dst are just for debugging, remove or improve when finished
__kernel void derivative_nonmax_suppress(
    __read_only  image2d_t src,
    __write_only image2d_t dst,
    __global const DerivInfo *deriv_buf,
    // .s0 (.x) is dx, .s1 (.y) is dy
    __global float2 *deriv_buf_suppressed
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    DerivInfo dinfo = read_from_1d_arrdinfo(deriv_buf, loc);
    float angle = dinfo.gradient_direction;
    bool is_max = false;

    // TODO: lots of redudant copies out of deriv_buf, could be improved
    if ((angle >= -22.5 && angle <= 22.5) || (angle < -157.5 && angle >= -180)) {
        // Compare with pixels left and right of loc
        if (
            dinfo.magnitude >= read_from_1d_arrdinfo(deriv_buf, (int2)(loc.x - 1, loc.y)).magnitude &&
            dinfo.magnitude >= read_from_1d_arrdinfo(deriv_buf, (int2)(loc.x + 1, loc.y)).magnitude
        ) {
            is_max = true;
        }
    } else if ((angle >= 22.5 && angle <= 67.5) || (angle < -112.5 && angle >= -157.5)) {
        // Compare with pixels diagonally top right and bottom left of loc
        if (
            dinfo.magnitude >= read_from_1d_arrdinfo(deriv_buf, (int2)(loc.x - 1, loc.y - 1)).magnitude &&
            dinfo.magnitude >= read_from_1d_arrdinfo(deriv_buf, (int2)(loc.x + 1, loc.y + 1)).magnitude
        ) {
            is_max = true;
        }
    } else if ((angle >= 67.5 && angle <= 112.5) || (angle < -67.5 && angle >= -112.5)) {
        // Compare with pixels above and below loc
        if (
            dinfo.magnitude >= read_from_1d_arrdinfo(deriv_buf, (int2)(loc.x, loc.y - 1)).magnitude &&
            dinfo.magnitude >= read_from_1d_arrdinfo(deriv_buf, (int2)(loc.x, loc.y + 1)).magnitude
        ) {
            is_max = true;
        }
    } else if ((angle >= 112.5 && angle <= 157.5) || (angle < -22.5 && angle >= -67.5)) {
        // Compare with pixels diagonally top left and bottom right of loc
        if (
            dinfo.magnitude >= read_from_1d_arrdinfo(deriv_buf, (int2)(loc.x + 1, loc.y - 1)).magnitude &&
            dinfo.magnitude >= read_from_1d_arrdinfo(deriv_buf, (int2)(loc.x - 1, loc.y + 1)).magnitude
        ) {
            is_max = true;
        }
    }

    if (is_max) {
        write_to_1d_arrf2(deriv_buf_suppressed, loc, (float2)(dinfo.dx, dinfo.dy));
        write_imagef(dst, loc, dinfo.magnitude);
    } else {
        write_to_1d_arrf2(deriv_buf_suppressed, loc, (float2)(0));
        write_imagef(dst, loc, (float4)(0.0f, 0.0f, 0.0f, 1.0f));
    }
}

// This kernel computes the harris response for the given src image from the given
// deriv_buf_suppressed and writes it to harris_buf
// TODO: src and dst are just for debugging, remove or improve when finished
__kernel void harris_response(
    __read_only  image2d_t src,
    __write_only image2d_t dst,
    __global const float2 *deriv_buf_suppressed,
    __global float *harris_buf
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    float sumdxdy = sum_deriv_prod(deriv_buf_suppressed, loc);
    float sumdx2 = sum_deriv_pow(deriv_buf_suppressed, loc, true);
    float sumdy2 = sum_deriv_pow(deriv_buf_suppressed, loc, false);

    float trace = sumdx2 + sumdy2;
    // r = det(M) - k(trace(M))^2
    // k usually between 0.04 to 0.06
    // threshold around 5?
    float r = (sumdx2 * sumdy2 - pow(sumdxdy, 2)) - 0.04f * (trace * trace);

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
}

// Performs non-maximum suppression on the given buffer (buffer is expected to
// represent harris response for an image) and writes the output to another.
//
// This means that for each response value, it checks all
// surrounding values within a hardcoded window and rejects the value if it is
// not the largest within that window.
// TODO: src and dst are just for debugging, remove or improve when finished
__kernel void harris_nonmax_suppress(
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
    // write_imagef(dst, loc, pixel);
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
    __global PointPair *matches_buf,
    int search_radius
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    ulong8 desc = read_from_1d_arrul8(desc_buf, loc);

    // TODO: restructure data so we don't have to do this
    // also this is an ugly hack
    if (desc.s0 == 0 && desc.s1 == 0) {
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
                write_to_1d_arrpointp(
                    matches_buf,
                    loc,
                    (PointPair) {
                        prev_point,
                        loc
                    }
                );

                // Debug stuff
                // Green box: point that was matched to a point in the previous frame
                draw_box(dst, loc, (float4)(0.0f, 1.0f, 0.0f, 1.0f), 5);
                // Blue box: said point in previous frame
                draw_box(dst, prev_point, (float4)(0.0f, 0.0f, 1.0f, 1.0f), 3);

                return;
            }
        }
    }

    if (has_compared == false) {
        // Red box: point that has nothing to compare to in the previous frame
        // draw_box(dst, loc, (float4)(1.0f, 0.0f, 0.0f, 1.0f), 5);
    }
}
