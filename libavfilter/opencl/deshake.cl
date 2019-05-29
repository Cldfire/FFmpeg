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

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE |
                          CLK_FILTER_NEAREST;

// Returns the luminance value at the given point.
float luminance(image2d_t src, int2 loc) {
    float4 pixel = read_imagef(src, sampler, loc);
    return (pixel.x + pixel.y + pixel.z) / 3.0f;
}

float prewitt(image2d_t src, int2 loc, int mask[3][3]) {
    return mask[0][0] * luminance(src, (int2)(loc.x - 1, loc.y + 1)) +
           mask[0][1] * luminance(src, (int2)(loc.x, loc.y + 1)) +
           mask[0][2] * luminance(src, (int2)(loc.x + 1, loc.y + 1)) +
           mask[1][0] * luminance(src, (int2)(loc.x - 1, loc.y)) +
           mask[1][1] * luminance(src, (int2)(loc.x, loc.y)) +
           mask[1][2] * luminance(src, (int2)(loc.x + 1, loc.y)) +
           mask[2][0] * luminance(src, (int2)(loc.x - 1, loc.y - 1)) +
           mask[2][1] * luminance(src, (int2)(loc.x, loc.y - 1)) +
           mask[2][2] * luminance(src, (int2)(loc.x + 1, loc.y - 1));
}

__kernel void deshake(
    __read_only  image2d_t src,
    __write_only image2d_t dst
) {
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef(src, sampler, loc);

    int mask_x[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };

    int mask_y[3][3] = {
        {1, 1, 1},
        {0, 0, 0},
        {-1, -1, -1}
    };

    float dx = prewitt(src, loc, mask_x);
    float dy = prewitt(src, loc, mask_y);

    pixel.x = pixel.y = pixel.z = sqrt(dx * dx + dy * dy);
    write_imagef(dst, loc, pixel);
}
