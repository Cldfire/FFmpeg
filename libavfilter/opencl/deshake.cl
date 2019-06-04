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

// TODO: is there a way to define these in one file?
#define BRIEFN 512

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

// Sums dx * dy for all pixels surrounding loc
float sum_deriv_prod(image2d_t src, int2 loc, int mask_x[3][3], int mask_y[3][3]) {
    float ret = 0;

    // These loops touch each pixel surrounding loc as well as loc itself
    for (int i = 1; i >= -1; --i) {
        for (int j = -1; j <= 1; ++j) {
            ret += convolve(src, (int2)(loc.x + j, loc.y + i), mask_x) *
                   convolve(src, (int2)(loc.x + j, loc.y + i), mask_y);
        }
    }

    return ret;
}

// Sums d<>^2 (determined by mask) for all pixels surrounding loc
float sum_deriv_pow(image2d_t src, int2 loc, int mask[3][3]) {
    float ret = 0;

    // These loops touch each pixel surrounding loc as well as loc itself
    for (int i = 1; i >= -1; --i) {
        for (int j = -1; j <= 1; ++j) {
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

// Writes to a 1D array at loc, treating it as a 2D array with the same
// dimensions as the global work size.
void write_float_to_1d_arr(__global float *buf, int2 loc, float val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

void write_ulong_to_1d_arr(__global ulong *buf, int2 loc, ulong val) {
    buf[loc.x + loc.y * get_global_size(0)] = val;
}

// Above except reading
float read_from_1d_arr(__global float *buf, int2 loc) {
    return buf[loc.x + loc.y * get_global_size(0)];
}

// This kernel computes the harris response for the given src image and writes
// it to harris_buf
// TODO: src and dst are just for debugging, remove or improve when finished
__kernel void harris_response(
    __read_only  image2d_t src,
    __write_only image2d_t dst,
    __global __write_only float *harris_buf
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

    // float sumdxdy = sum_deriv_prod(src, loc, sobel_mask_x, sobel_mask_y);
    float sumdx2 = sum_deriv_pow(src, loc, sobel_mask_x);
    float sumdy2 = sum_deriv_pow(src, loc, sobel_mask_y);

    // float trace = sumdx2 + sumdy2;
    // r = det(M) - k(trace(M))^2
    // k usually between 0.04 to 0.06
    // threshold around 5?
    // float r = (sumdx2 * sumdy2 - pow(sumdxdy, 2)) - 0.04f * (trace * trace);

    // This is the shi-tomasi method of calculating r
    // threshold around 2.5?
    float r = min(sumdx2, sumdy2);

    // Threshold the r value
    if (r > 3.0f) {
        // draw_box(dst, loc, (float4)(0.0f, 1.0f, 0.0f, 1.0f), 5);
        write_float_to_1d_arr(harris_buf, loc, r);
    } else {
        write_float_to_1d_arr(harris_buf, loc, 0.0f);
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
    const int window_size = 7;
    const int half_window = window_size / 2;

    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float center_val = read_from_1d_arr(harris_buf, loc);

    if (center_val == 0.0f) {
        // obviously not a maximum
        write_float_to_1d_arr(harris_buf_suppressed, loc, center_val);
        goto done;
    }

    // TODO: could save an iteration by not comparing the center value to itself
    for (int i = -half_window; i <= half_window; ++i) {
        for (int j = -half_window; j <= half_window; ++j) {
            if (center_val < read_from_1d_arr(harris_buf, (int2)(loc.x + i, loc.y + j))) {
                // This value is not the maximum within the window
                write_float_to_1d_arr(harris_buf_suppressed, loc, 0.0f);
                goto done;
            }
        }
    }

    write_float_to_1d_arr(harris_buf_suppressed, loc, center_val);
    goto done;

done: // debug stuff
    // write_imagef(dst, loc, center_val, loc));
    
    if (center_val != 0.0f) {
        draw_box(dst, loc, (float4)(1.0f, 0.0f, 0.0f, 1.0f), 5);
    }
    float4 pixel = read_imagef(src, sampler, loc);
    write_imagef(dst, loc, pixel);
}

// Extracts BRIEF descriptors from the src image for the given features using the
// provided sampler.
__kernel void brief_descriptors(
    __read_only image2d_t src,
    // TODO: changing BRIEFN will make this a different type, figure out how to
    // deal with that
    __global const float *features,
    __global ulong *desc_buf,
    __global const PointPair *brief_sampler
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    // TODO: restructure data so we don't have to do this
    if (read_from_1d_arr(features, loc) == 0.0f) {
        return;
    }

    ulong desc = 0;
    for (int i = 0; i < BRIEFN; ++i) {
        PointPair pair = brief_sampler[i];
        float l1 = luminance(src, (int2)(loc.x + pair.p1.x, loc.y + pair.p1.y));
        float l2 = luminance(src, (int2)(loc.x + pair.p2.x, loc.y + pair.p2.y));

        if (l1 < l2) {
            desc |= 1UL << i;
        }
    }

    write_ulong_to_1d_arr(desc_buf, loc, desc);
}
