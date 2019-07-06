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

// TODO: probably shouldn't include this, does ffmpeg already have a rand utility?
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/fifo.h"
#include "avfilter.h"
#include "framequeue.h"
#include "filters.h"
#include "transform.h"
#include "formats.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"

// Block size over which to compute harris response
#define HARRIS_RADIUS 2
// Number of bits for BRIEF descriptors
// Choices: 128, 256, 512
#define BREIFN 512
// Size of the patch from which a BRIEF descriptor is extracted
// This is the size used in OpenCV
#define BRIEF_PATCH_SIZE 31
#define BRIEF_PATCH_SIZE_HALF BRIEF_PATCH_SIZE / 2
// The radius within which to search around descriptors for matches from the
// previous frame
// TODO: not sure what the optimal value is here
#define MATCH_SEARCH_RADIUS 70

#define MATCHES_CONTIG_SIZE 1000

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

#define sign(x)  ((signbit(x) ?  -1 : 1))

typedef struct PointPair {
    // Previous frame
    cl_int2 p1;
    // Current frame
    cl_int2 p2;
} PointPair;

typedef struct SmoothedPointPair {
    // Previous frame
    cl_int2 p1;
    // Smoothed point in current frame
    cl_float2 p2;
} SmoothedPointPair;

// TODO: should probably rename to MotionVector or something
typedef struct Vector {
    PointPair p;
    // Can be set to -1 to specify invalid vector
    float magnitude;
    // Angle is in degrees
    float angle;
    // Used to mark vectors as potential outliers
    bool should_consider;
} Vector;

// Groups together the ringbuffers that store absolute distortion / position values
// for each frame
typedef struct AbsoluteFrameMotion {
    // x translation
    AVFifoBuffer *x;
    // y translation
    AVFifoBuffer *y;
    AVFifoBuffer *rot;
    AVFifoBuffer *scale_x;
    AVFifoBuffer *scale_y;

    // Offset to get to the current frame being processed
    // (not in bytes)
    int curr_frame_offset;
} AbsoluteFrameMotion;

// Stores the translation, scale, rotation, and skew deltas between two frames
typedef struct FrameMotion {
    cl_float2 translation;
    float rotation;
    cl_float2 scale;
    cl_float2 skew;
} FrameMotion;

typedef struct SimilarityMatrix {
    // The 2x3 similarity matrix
    double matrix[6];
} SimilarityMatrix;

typedef struct CropInfo {
    // The top left corner of the bounding box for the crop
    cl_float2 top_left;
    // The bottom right corner of the bounding box for the crop
    cl_float2 bottom_right;
} CropInfo;

// Returned from function that determines start and end values for iteration
// around the current frame in a ringbuffer
typedef struct IterIndices {
    int start;
    int end;
} IterIndices;

typedef struct DeshakeOpenCLContext {
    OpenCLFilterContext ocf;
    // Whether or not the above `OpenCLFilterContext` has been initialized
    int initialized;

    // These variables are used in the activate callback
    int64_t duration;
    bool eof;

    // FIFO frame queue used to buffer future frames for processing
    FFFrameQueue fq;
    // Ringbuffers for frame positions
    AbsoluteFrameMotion abs_motion;

    // The number of frames' motion to consider before and after the frame we are
    // smoothing
    int smooth_window;
    // The number of the frame we are currently processing
    int curr_frame;

    // Stores a 1d array of normalised gaussian kernel values for convolution
    float *gauss_kernel;
    // Stores a kernel half the size of the above for use in determining which
    // sigma value to use for the above
    float *small_gauss_kernel;

    // Information regarding how to crop the smoothed frames
    CropInfo crop;

    // Buffer to copy `matches` into for the CPU to work with
    Vector *matches_host;
    Vector *matches_contig_host;

    cl_command_queue command_queue;
    cl_kernel kernel_harris;
    cl_kernel kernel_nonmax_suppress;
    cl_kernel kernel_brief_descriptors;
    cl_kernel kernel_match_descriptors;
    cl_kernel kernel_transform;
    cl_kernel kernel_crop_upscale;

    cl_kernel kernel_debug_matches;

    cl_mem harris_buf;
    // Harris response after non-maximum suppression
    cl_mem harris_buf_suppressed;

    // BRIEF sampling pattern that is randomly initialized
    cl_mem brief_pattern;

    // Feature point descriptors for the current frame
    cl_mem descriptors;
    // Feature point descriptors for the previous frame
    cl_mem prev_descriptors;
    // Vectors between points in current and previous frame
    cl_mem matches;
    cl_mem matches_contig;
    // Holds the similarity matrix to transform a frame with
    cl_mem matrix;
} DeshakeOpenCLContext;

// Returns a random uniformly-distributed number in [low, high]
static int rand_in(int low, int high) {
    double rand_val = rand() / (1.0 + RAND_MAX); 
    int range = high - low + 1;
    int rand_scaled = (rand_val * range) + low;

    return rand_scaled;
}

// The following code is loosely ported from OpenCV

// Estimates affine transform from 3 point pairs
// model is a 2x3 matrix:
//      a b c
//      d e f
static int run_estimate_kernel(const Vector *point_pairs, double *model) {
    // src points
    double x1 = point_pairs[0].p.p1.s[0];
    double y1 = point_pairs[0].p.p1.s[1];
    double x2 = point_pairs[1].p.p1.s[0];
    double y2 = point_pairs[1].p.p1.s[1];
    double x3 = point_pairs[2].p.p1.s[0];
    double y3 = point_pairs[2].p.p1.s[1];

    // dest points
    double X1 = point_pairs[0].p.p2.s[0];
    double Y1 = point_pairs[0].p.p2.s[1];
    double X2 = point_pairs[1].p.p2.s[0];
    double Y2 = point_pairs[1].p.p2.s[1];
    double X3 = point_pairs[2].p.p2.s[0];
    double Y3 = point_pairs[2].p.p2.s[1];

    double d = 1.0 / ( x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2) );

    model[0] = d * ( X1*(y2-y3) + X2*(y3-y1) + X3*(y1-y2) );
    model[1] = d * ( X1*(x3-x2) + X2*(x1-x3) + X3*(x2-x1) );
    model[2] = d * ( X1*(x2*y3 - x3*y2) + X2*(x3*y1 - x1*y3) + X3*(x1*y2 - x2*y1) );

    model[3] = d * ( Y1*(y2-y3) + Y2*(y3-y1) + Y3*(y1-y2) );
    model[4] = d * ( Y1*(x3-x2) + Y2*(x1-x3) + Y3*(x2-x1) );
    model[5] = d * ( Y1*(x2*y3 - x3*y2) + Y2*(x3*y1 - x1*y3) + Y3*(x1*y2 - x2*y1) );

    return 1;
}

// Checks a subset of 3 point pairs to make sure that the points are not collinear
// and not too close to each other
static bool check_subset(const Vector *pairs_subset) {
    int j, k, i = 2;

    // TODO: make point struct and split this into points_are_collinear func
    for (j = 0; j < i; j++) {
        double dx1 = pairs_subset[j].p.p1.s[0] - pairs_subset[i].p.p1.s[0];
        double dy1 = pairs_subset[j].p.p1.s[1] -pairs_subset[i].p.p1.s[1];

        for (k = 0; k < j; k++) {
            double dx2 = pairs_subset[k].p.p1.s[0] - pairs_subset[i].p.p1.s[0];
            double dy2 = pairs_subset[k].p.p1.s[1] - pairs_subset[i].p.p1.s[1];

            if (fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2))) {
                return false;
            }
        }
    }

    for (j = 0; j < i; j++) {
        double dx1 = pairs_subset[j].p.p2.s[0] - pairs_subset[i].p.p2.s[0];
        double dy1 = pairs_subset[j].p.p2.s[1] -pairs_subset[i].p.p2.s[1];

        for (k = 0; k < j; k++) {
            double dx2 = pairs_subset[k].p.p2.s[0] - pairs_subset[i].p.p2.s[0];
            double dy2 = pairs_subset[k].p.p2.s[1] - pairs_subset[i].p.p2.s[1];

            if (fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2))) {
                return false;
            }
        }
    }

    return true;
}

// Selects a random subset of 3 points from point_pairs and places them in pairs_subset
static bool get_subset(
    const Vector *point_pairs,
    const int num_point_pairs,
    Vector *pairs_subset,
    int max_attempts
) {
    int idx[3];
    int i = 0, j, iters = 0;

    for (; iters < max_attempts; iters++) {
        for (i = 0; i < 3 && iters < max_attempts;) {
            int idx_i = 0;

            for (;;) {
                idx_i = idx[i] = rand_in(0, num_point_pairs);

                for (j = 0; j < i; j++) {
                    if (idx_i == idx[j]) {
                        break;
                    }
                }

                if (j == i) {
                    break;
                }
            }

            pairs_subset[i] = point_pairs[idx[i]];
            i++;
        }

        if (i == 3 && !check_subset(pairs_subset)) {
            continue;
        }
        break;
    }

    return i == 3 && iters < max_attempts;
}

// Computes the error for each of the given points based on the given model.
static void compute_error(
    const Vector *point_pairs,
    const int num_point_pairs,
    const double *model,
    float *err
) {
    float F0 = (float)model[0], F1 = (float)model[1], F2 = (float)model[2];
    float F3 = (float)model[3], F4 = (float)model[4], F5 = (float)model[5];

    for (int i = 0; i < num_point_pairs; i++) {
        const cl_int2 *f = &point_pairs[i].p.p1;
        const cl_int2 *t = &point_pairs[i].p.p2;

        float a = F0*f->s[0] + F1*f->s[1] + F2 - t->s[0];
        float b = F3*f->s[0] + F4*f->s[1] + F5 - t->s[1];

        err[i] = a*a + b*b;
    }
}

// Determines which of the given point matches are inliers for the given model
// based on the specified threshold.
//
// err must be an array of num_point_pairs length
static int find_inliers(
    Vector *point_pairs,
    const int num_point_pairs,
    const double *model,
    float *err,
    double thresh
) {
    float t = (float)(thresh * thresh);
    int i, n = num_point_pairs, num_inliers = 0;

    compute_error(point_pairs, num_point_pairs, model, err);

    for (i = 0; i < n; i++) {
        if (err[i] <= t) {
            // This is an inlier
            point_pairs[i].should_consider = true;
            num_inliers += 1;
        } else {
            point_pairs[i].should_consider = false;
        }
    }

    return num_inliers;
}

static int ransac_update_num_iters(double p, double ep, int max_iters) {
    double num, denom;

    // TODO: replace with actual clamping code
    p = MAX(p, 0.0);
    p = MIN(p, 1.0);
    ep = MAX(ep, 0.0);
    ep = MIN(ep, 1.0);

    // avoid inf's & nan's
    num = MAX(1.0 - p, DBL_MIN);
    denom = 1.0 - pow(1.0 - ep, 3);
    if (denom < DBL_MIN) {
        return 0;
    }

    num = log(num);
    denom = log(denom);

    // TODO: opencv uses cvround, make sure it doesn't do anything special
    return denom >= 0 || -num >= max_iters * (-denom) ? max_iters : (int)round(num / denom);
}

// Determines inliers from the given pairs of points using RANdom SAmple Consensus.
static bool runRansacPointSetRegistrator(
    Vector *point_pairs,
    const int num_point_pairs,
    double *model_out,
    const double threshold,
    const double confidence,
    const int max_iters
) {
    bool result = false;
    double best_model[6], model[6];
    Vector pairs_subset[3];
    float *err;
    int nmodels;

    int iter, niters = MAX(max_iters, 1);
    int good_count, max_good_count = 0;

    // We need at least 3 points to build a model from
    if (num_point_pairs < 3) {
        return false;
    } else if (num_point_pairs == 3) {
        // There are only 3 points, so RANSAC doesn't apply here
        if (run_estimate_kernel(point_pairs, model_out) <= 0) {
            return false;
        }

        for (int i = 0; i < 3; ++i) {
            point_pairs[i].should_consider = true;
        }

        return true;
    }

    err = av_malloc_array((size_t)num_point_pairs, sizeof(float));
    if (!err) {
        // TODO: do something better here
        return false;
    }

    for (iter = 0; iter < niters; ++iter) {
        // TODO: didn't we already catch this case above?
        if (num_point_pairs > 3) {
            bool found = get_subset(point_pairs, num_point_pairs, pairs_subset, 10000);

            if (!found) {
                if (iter == 0) {
                    return false;
                }

                break;
            }
        }

        nmodels = run_estimate_kernel(pairs_subset, model);
        // TODO: probably don't need this (or nmodels at all)
        if (nmodels <= 0) {
            continue;
        }

        // The original code is written to deal with multiple model estimations for
        // a single trio of points, but we will always only have 1

        good_count = find_inliers(point_pairs, num_point_pairs, model, err, threshold);

        if (good_count > MAX(max_good_count, 2)) {
            for (int mi = 0; mi < 6; ++mi) {
                best_model[mi] = model[mi];
            }

            max_good_count = good_count;
            niters = ransac_update_num_iters(
                confidence,
                (double)(num_point_pairs - good_count) / num_point_pairs,
                niters
            );
        }
    }

    if (max_good_count > 0) {
        for (int mi = 0; mi < 6; ++mi) {
            model_out[mi] = best_model[mi];
        }
        find_inliers(point_pairs, num_point_pairs, best_model, err, threshold);

        result = true;
    }

    av_free(err);
    return result;
}

// Estimates an affine transform between the given pairs of points using RANSAC.
static bool estimate_affine_2d(
    Vector *point_pairs,
    const int num_point_pairs,
    double *model_out,
    const double ransac_reproj_threshold,
    const int max_iters,
    const double confidence
) {
    bool result = false;

    result = runRansacPointSetRegistrator(
        point_pairs,
        num_point_pairs,
        model_out,
        ransac_reproj_threshold,
        confidence,
        max_iters
    );

    // could do levmarq here to refine the transform
    // but levmarq is very complicated and that shouldn't be necessary anyway

    return result;
}

// End code from OpenCV

// Decomposes a similarity matrix into translation, rotation, scale, and skew
//
// See http://frederic-wang.fr/decomposition-of-2d-transform-matrices.html
static FrameMotion decompose_transform(double *model) {
    FrameMotion ret;

    double a = model[0];
    double c = model[1];
    double e = model[2];
    double b = model[3];
    double d = model[4];
    double f = model[5];
    double delta = a * d - b * c;

    ret.translation.s[0] = e;
    ret.translation.s[1] = f;

    // This is the QR method
    if (a != 0 || b != 0) {
        double r = sqrt(a * a + b * b);

        ret.rotation = sign(b) * acos(a / r);
        ret.scale.s[0] = r;
        ret.scale.s[1] = delta / r;
        ret.skew.s[0] = atan((a * c + b * d) / (r * r));
        ret.skew.s[1] = 0;
    } else if (c != 0 || d != 0) {
        double s = sqrt(c * c + d * d);

        ret.rotation = M_PI / 2 - sign(d) * acos(-c / s);
        ret.scale.s[0] = delta / s;
        ret.scale.s[1] = s;
        ret.skew.s[0] = 0;
        ret.skew.s[1] = atan((a * c + b * d) / (s * s));
    } // otherwise there is only translation

    return ret;
}

// Read from a 1d array representing a value for each pixel of a frame given
// the necessary details
static Vector read_from_1d_arrvec(const Vector *buf, int width, int x, int y) {
    return buf[x + y * width];
}

// Move valid vectors from the 2d buffer into a 1d buffer where they are contiguous
static int make_vectors_contig(
    DeshakeOpenCLContext *deshake_ctx,
    int frame_height,
    int frame_width
) {
    int num_vectors = 0;

    for (int i = 0; i < frame_height; ++i) {
        for (int j = 0; j < frame_width; ++j) {
            Vector v = read_from_1d_arrvec(deshake_ctx->matches_host, frame_width, j, i);

            if (v.should_consider) {
                deshake_ctx->matches_contig_host[num_vectors] = v;
                ++num_vectors;
            }

            // Make sure we do not exceed the amount of space we allocated for these vectors
            if (num_vectors == MATCHES_CONTIG_SIZE - 1) {
                return num_vectors;
            }
        }
    }
    return num_vectors;
}

// Returns the gaussian kernel value for the given x coordinate and sigma value
static float gaussian_for(int x, float sigma) {
    return 1.0f / powf(M_E, ((float)x * (float)x) / (2.0f * sigma * sigma));
}

// Makes a normalized gaussian kernel of the given length for the given sigma
// and places it in gauss_kernel
static void make_gauss_kernel(float *gauss_kernel, float length, float sigma) {
    float gauss_sum = 0;
    int window_half = length / 2;

    for (int i = 0; i < length; ++i) {
        float val = gaussian_for(i - window_half, sigma);

        gauss_sum += val;
        gauss_kernel[i] = val;
    }

    // Normalize the gaussian values
    for (int i = 0; i < length; ++i) {
        gauss_kernel[i] /= gauss_sum;
    }
}

// Returns indices to start and end iteration at in order to iterate over a window
// of length size centered at the current frame in a ringbuffer
//
// Always returns numbers that result in a window of length size, even if that
// means specifying negative indices or indices past the end of the values in the
// ringbuffers. Make sure you clip indices appropriately within your loop.
static IterIndices start_end_for(DeshakeOpenCLContext *deshake_ctx, int length) {
    IterIndices indices;

    // TODO: make this use length
    if (deshake_ctx->abs_motion.curr_frame_offset < length / 2) {
        // Start of video; repeat first frame data
        indices.start = deshake_ctx->abs_motion.curr_frame_offset - (length / 2);
        indices.end = deshake_ctx->abs_motion.curr_frame_offset + (length / 2) - 1;
    } else {
        // Somewhere else in video, all data available
        // Or end, in which case the clipping in the loop handles repeating last frame data
        indices.start = 0;
        indices.end = length - 1;
    }

    return indices;
}

// Sets val to the value in the given ringbuffer at the given offset, taking care of
// clipping the offset into the appropriate range
static void ringbuf_float_at(
    DeshakeOpenCLContext *deshake_ctx, 
    AVFifoBuffer *values,
    float *val,
    int offset
) {
        int offset_clipped = av_clip(
            offset,
            // Negative indices will occur at the start of the video, and we want
            // them to be clipped to 0 in order to repeatedly use the position of
            // the first frame.
            0, 
            // This expression represents the last valid index in the buffer,
            // which we use repeatedly at the end of the video.
            deshake_ctx->smooth_window - (av_fifo_space(values) / sizeof(float)) - 1
        );

        av_fifo_generic_peek_at(
            values,
            val,
            offset_clipped * sizeof(float),
            sizeof(float),
            NULL
        );
}

// Returns smoothed current frame value of the given buffer of floats based on the
// given Gaussian kernel and its length (also the window length, centered around the
// current frame) and the "maximum value" of the motion.
//
// This "maximum value" should be the width / height of the image in the case of
// translation and an empirically chosen constant for rotation / scale
static float smooth(
    DeshakeOpenCLContext *deshake_ctx,
    float *gauss_kernel,
    int length,
    float max_val,
    AVFifoBuffer *values
) {
    float new_large_s = 0, new_small_s = 0, new_best = 0, old, diff_between,
          percent_of_max, inverted_percent;
    IterIndices indices = start_end_for(deshake_ctx, length);
    float large_sigma = 40.0f;
    float small_sigma = 2.0f;
    float best_sigma;

    // Strategy to adaptively smooth trajectory:
    //
    // 1. Smooth path with large and small sigma values
    // 2. Take the absolute value of the difference between them
    // 3. Get a percentage by putting the difference over the "max value"
    // 4, Invert the percentage
    // 5. Calculate a new sigma value weighted towards the larger sigma value
    // 6. Determine final smoothed trajectory value using that sigma

    make_gauss_kernel(gauss_kernel, length, large_sigma);
    for (int i = indices.start, j = 0; i < indices.end; ++i, ++j) {
        ringbuf_float_at(deshake_ctx, values, &old, i);
        new_large_s += old * gauss_kernel[j];
    }

    make_gauss_kernel(gauss_kernel, length, small_sigma);
    for (int i = indices.start, j = 0; i < indices.end; ++i, ++j) {
        ringbuf_float_at(deshake_ctx, values, &old, i);
        new_small_s += old * gauss_kernel[j];
    }

    diff_between = fabsf(new_large_s - new_small_s);
    percent_of_max = diff_between / max_val;
    inverted_percent = 1 - percent_of_max;
    best_sigma = large_sigma * powf(inverted_percent, 40);

    printf("best sigma: %f\n", best_sigma);
    printf("        percent_of_max was %f. inverted_percent was %f.\n", percent_of_max, inverted_percent);

    make_gauss_kernel(gauss_kernel, length, best_sigma);
    for (int i = indices.start, j = 0; i < indices.end; ++i, ++j) {
        ringbuf_float_at(deshake_ctx, values, &old, i);
        new_best += old * gauss_kernel[j];
    }

    return new_best;
}

// TODO: should this be merged with `avfilter_get_matrix`?
static void affine_transform_matrix(
    float x_shift,
    float y_shift,
    float angle,
    float scale_x,
    float scale_y,
    float *matrix
) {
    matrix[0] = scale_x * cos(angle);
    matrix[1] = -sin(angle);
    matrix[2] = x_shift;
    matrix[3] = -matrix[1];
    matrix[4] = scale_y * cos(angle);
    matrix[5] = y_shift;
    matrix[6] = 0;
    matrix[7] = 0;
    matrix[8] = 1;
}

// Returns the position of the given point after the transform is applied
static cl_float2 transformed_point(float x, float y, float *transform) {
    cl_float2 ret;

    ret.s[0] = x * transform[0] + y * transform[1] + transform[2];
    ret.s[1] = x * transform[3] + y * transform[4] + transform[5];

    return ret;
}

// Determines the crop necessary to eliminate black borders from a smoothed frame
// and updates target crop accordingly
static void update_needed_crop(
    DeshakeOpenCLContext *deshake_ctx,
    float *transform,
    float frame_width,
    float frame_height
) {
    float new_width, new_height, adjusted_width, adjusted_height, adjusted_x, adjusted_y;

    cl_float2 top_left = transformed_point(0, 0, transform);
    cl_float2 top_right = transformed_point(frame_width, 0, transform);
    cl_float2 bottom_left = transformed_point(0, frame_height, transform);
    cl_float2 bottom_right = transformed_point(frame_width, frame_height, transform);
    float ar_h = frame_height / frame_width;
    float ar_w = frame_width / frame_height;
    CropInfo *crop = &deshake_ctx->crop;

    crop->top_left.s[0] = MAX(
        crop->top_left.s[0],
        MAX(
            top_left.s[0],
            bottom_left.s[0]
        )
    );

    crop->top_left.s[1] = MAX(
        crop->top_left.s[1],
        MAX(
            top_left.s[1],
            top_right.s[1]
        )
    );

    crop->bottom_right.s[0] = MIN(
        crop->bottom_right.s[0],
        MIN(
            bottom_right.s[0],
            top_right.s[0]
        )
    );

    crop->bottom_right.s[1] = MIN(
        crop->bottom_right.s[1],
        MIN(
            bottom_right.s[1],
            bottom_left.s[1]
        )
    );

    // Make sure our potentially new bounding box has the same aspect ratio
    new_height = crop->bottom_right.s[1] - crop->top_left.s[1];
    new_width = crop->bottom_right.s[0] - crop->top_left.s[0];

    adjusted_width = new_height * ar_w;
    adjusted_x = crop->bottom_right.s[0] - adjusted_width;

    if (adjusted_x >= crop->top_left.s[0]) {
        crop->top_left.s[0] = adjusted_x;
    } else {
        adjusted_height = new_width * ar_h;
        adjusted_y = crop->bottom_right.s[1] - adjusted_height;
        crop->top_left.s[1] = adjusted_y;
    }
}

// Allocates ringbuffers with the appropriate amount of space for the necessary
// motion information
//
// Returns 0 if all allocations were successful or AVERROR(ENOMEM) if any failed
static int init_abs_motion_ringbuffers(DeshakeOpenCLContext *deshake_ctx) {
    deshake_ctx->abs_motion.x = av_fifo_alloc_array(
        deshake_ctx->smooth_window,
        sizeof(float)
    );

    if (!deshake_ctx->abs_motion.x) {
        return AVERROR(ENOMEM);
    }

    deshake_ctx->abs_motion.y = av_fifo_alloc_array(
        deshake_ctx->smooth_window,
        sizeof(float)
    );

    if (!deshake_ctx->abs_motion.y) {
        return AVERROR(ENOMEM);
    }

    deshake_ctx->abs_motion.rot = av_fifo_alloc_array(
        deshake_ctx->smooth_window,
        sizeof(float)
    );

    if (!deshake_ctx->abs_motion.rot) {
        return AVERROR(ENOMEM);
    }

    deshake_ctx->abs_motion.scale_x = av_fifo_alloc_array(
        deshake_ctx->smooth_window,
        sizeof(float)
    );

    if (!deshake_ctx->abs_motion.scale_x) {
        return AVERROR(ENOMEM);
    }

    deshake_ctx->abs_motion.scale_y = av_fifo_alloc_array(
        deshake_ctx->smooth_window,
        sizeof(float)
    );

    if (!deshake_ctx->abs_motion.scale_y) {
        return AVERROR(ENOMEM);
    }

    return 0;
}

// Frees the various ringbuffers used to store absolute motion information
static void uninit_abs_motion_ringbuffers(DeshakeOpenCLContext *deshake_ctx) {
    av_fifo_freep(&deshake_ctx->abs_motion.x);
    av_fifo_freep(&deshake_ctx->abs_motion.y);
    av_fifo_freep(&deshake_ctx->abs_motion.rot);
    av_fifo_freep(&deshake_ctx->abs_motion.scale_x);
    av_fifo_freep(&deshake_ctx->abs_motion.scale_y);
}

static int deshake_opencl_init(AVFilterContext *avctx, int frame_width, int frame_height)
{
    DeshakeOpenCLContext *ctx = avctx->priv;
    // Pointer to the host-side pattern buffer to be initialized and then copied
    // to the GPU
    PointPair *pattern_host;
    cl_int cle;
    int err;
    cl_ulong8 zeroed_ulong8;
    FFFrameQueueGlobal fqg;

    const int harris_buf_size = frame_height * frame_width * sizeof(float);
    const int descriptor_buf_size = frame_height * frame_width * (BREIFN / 8);

    ff_framequeue_global_init(&fqg);
    ff_framequeue_init(&ctx->fq, &fqg);
    ctx->eof = false;
    ctx->smooth_window = (int)(av_q2d(avctx->inputs[0]->frame_rate) * 2.0);
    ctx->curr_frame = 0;

    srand(947247);
    memset(&zeroed_ulong8, 0, sizeof(cl_ulong8));

    ctx->gauss_kernel = av_malloc_array(ctx->smooth_window, sizeof(float));
    if (!ctx->gauss_kernel) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    ctx->small_gauss_kernel = av_malloc_array(ctx->smooth_window / 2, sizeof(float));
    if (!ctx->small_gauss_kernel) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    ctx->crop.top_left.s[0] = 0;
    ctx->crop.top_left.s[1] = 0;
    ctx->crop.bottom_right.s[0] = frame_width;
    ctx->crop.bottom_right.s[1] = frame_height;

    err = init_abs_motion_ringbuffers(ctx);
    if (err < 0) {
        goto fail;
    }
    ctx->abs_motion.curr_frame_offset = 0;

    pattern_host = av_malloc_array(BREIFN, sizeof(PointPair));
    if (!pattern_host) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    ctx->matches_host = av_malloc_array(frame_height * frame_width, sizeof(Vector));
    if (!ctx->matches_host) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    ctx->matches_contig_host = av_malloc_array(MATCHES_CONTIG_SIZE, sizeof(Vector));
    if (!ctx->matches_contig_host) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    // Initializing the patch pattern for building BREIF descriptors with
    for (int i = 0; i < BREIFN; ++i) {
        PointPair pair;
        
        for (int j = 0; j < 2; ++j) {
            pair.p1.s[j] = rand_in(-BRIEF_PATCH_SIZE_HALF, BRIEF_PATCH_SIZE_HALF + 1);
            pair.p2.s[j] = rand_in(-BRIEF_PATCH_SIZE_HALF, BRIEF_PATCH_SIZE_HALF + 1);
        }

        pattern_host[i] = pair;
    }

    err = ff_opencl_filter_load_program(avctx, &ff_opencl_source_deshake, 1);
    if (err < 0)
        goto fail;

    ctx->command_queue = clCreateCommandQueue(
        ctx->ocf.hwctx->context,
        ctx->ocf.hwctx->device_id,
        0,
        &cle
    );

    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create OpenCL command queue %d.\n", cle);
    
    ctx->kernel_harris = clCreateKernel(ctx->ocf.program, "harris_response", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create harris_response kernel: %d.\n", cle);

    ctx->kernel_nonmax_suppress = clCreateKernel(ctx->ocf.program, "nonmax_suppression", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create nonmax_suppression kernel: %d.\n", cle);

    ctx->kernel_brief_descriptors = clCreateKernel(ctx->ocf.program, "brief_descriptors", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel_brief_descriptors kernel: %d.\n", cle);

    ctx->kernel_match_descriptors = clCreateKernel(ctx->ocf.program, "match_descriptors", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel_match_descriptors kernel: %d.\n", cle);

    ctx->kernel_transform = clCreateKernel(ctx->ocf.program, "transform", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel_transform kernel: %d.\n", cle);

    ctx->kernel_crop_upscale = clCreateKernel(ctx->ocf.program, "crop_upscale", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel_crop_upscale kernel: %d.\n", cle);

    ctx->kernel_debug_matches = clCreateKernel(ctx->ocf.program, "debug_matches", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel_debug_matches kernel: %d.\n", cle);

    // TODO: reduce boilerplate for creating buffers
    ctx->harris_buf = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        harris_buf_size,
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create harris_buf buffer: %d.\n", cle);

    ctx->harris_buf_suppressed = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        harris_buf_size,
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create harris_buf_suppressed buffer: %d.\n", cle);

    ctx->brief_pattern = clCreateBuffer(
        ctx->ocf.hwctx->context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        BREIFN * sizeof(PointPair),
        pattern_host,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create brief_pattern buffer: %d.\n", cle);

    ctx->descriptors = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        descriptor_buf_size,
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create descriptors buffer: %d.\n", cle);

    ctx->prev_descriptors = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        descriptor_buf_size,
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create prev_descriptors buffer: %d.\n", cle);

    cle = clEnqueueFillBuffer(
        ctx->command_queue,
        ctx->prev_descriptors,
        &zeroed_ulong8,
        sizeof(cl_ulong8),
        0,
        descriptor_buf_size,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue filling prev_desciptors buffer: %d.\n", cle);

    cle = clFinish(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue filling prev_descriptors buffer: %d.\n", cle);

    // TODO: don't need anywhere near this much memory allocated for this buffer
    ctx->matches = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        frame_height * frame_width * sizeof(Vector),
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create matches buffer: %d.\n", cle);

    ctx->matches_contig = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        MATCHES_CONTIG_SIZE * sizeof(Vector),
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create matches_contig buffer: %d.\n", cle);

    ctx->matrix = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        6 * sizeof(float),
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create matrix buffer: %d.\n", cle);

    ctx->initialized = 1;
    av_free(pattern_host);

    return 0;

fail:
    // TODO: see if it's possible to call the uninit function instead of all this
    // fq init happens before anything that could jump to this label
    ff_framequeue_free(&ctx->fq);
    uninit_abs_motion_ringbuffers(ctx);
    if (ctx->gauss_kernel)
        av_free(ctx->gauss_kernel);
    if (ctx->small_gauss_kernel)
        av_free(ctx->small_gauss_kernel);
    if (pattern_host)
        av_free(pattern_host);
    if (ctx->matches_host)
        av_free(ctx->matches_host);
    if (ctx->matches_contig_host)
        av_free(ctx->matches_contig_host);
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel_harris)
        clReleaseKernel(ctx->kernel_harris);
    if (ctx->kernel_nonmax_suppress)
        clReleaseKernel(ctx->kernel_nonmax_suppress);
    if (ctx->kernel_brief_descriptors)
        clReleaseKernel(ctx->kernel_brief_descriptors);
    if (ctx->kernel_match_descriptors)
        clReleaseKernel(ctx->kernel_match_descriptors);
    if (ctx->kernel_transform)
        clReleaseKernel(ctx->kernel_transform);
    if (ctx->kernel_crop_upscale)
        clReleaseKernel(ctx->kernel_crop_upscale);
    if (ctx->kernel_debug_matches)
        clReleaseKernel(ctx->kernel_debug_matches);
    if (ctx->harris_buf)
        clReleaseMemObject(ctx->harris_buf);
    if (ctx->harris_buf_suppressed)
        clReleaseMemObject(ctx->harris_buf_suppressed);
    if (ctx->brief_pattern)
        clReleaseMemObject(ctx->brief_pattern);
    if (ctx->descriptors)
        clReleaseMemObject(ctx->descriptors);
    if (ctx->prev_descriptors)
        clReleaseMemObject(ctx->prev_descriptors);
    if (ctx->matches)
        clReleaseMemObject(ctx->matches);
    if (ctx->matches_contig)
        clReleaseMemObject(ctx->matches_contig);
    if (ctx->matrix)
        clReleaseMemObject(ctx->matrix);
    return err;
}

// Uses the buffered motion information to determine a transform that smooths the
// given frame and applies it
static int filter_frame(AVFilterLink *link, AVFrame *input_frame) {
    AVFilterContext *avctx = link->dst;
    AVFilterLink *outlink = avctx->outputs[0];
    DeshakeOpenCLContext *deshake_ctx = avctx->priv;
    AVFrame *output_frame = NULL, *transformed_frame = NULL;
    int err;
    cl_int cle;
    float new_x, new_y, new_rot, new_scale_x, new_scale_y, old_x, old_y,
          old_rot, old_scale_x, old_scale_y;
    float transform[9];
    size_t global_work[2];
    int64_t duration;
    cl_mem src, transformed, dst;

    if (input_frame->pkt_duration) {
        duration = input_frame->pkt_duration;
    } else {
        duration = av_rescale_q(1, av_inv_q(outlink->frame_rate), outlink->time_base);
    }
    deshake_ctx->duration = input_frame->pts + duration;

    // Get the absolute transform data for this frame
    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.x,
        &old_x,
        deshake_ctx->abs_motion.curr_frame_offset * sizeof(float),
        sizeof(float),
        NULL
    );

    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.y,
        &old_y,
        deshake_ctx->abs_motion.curr_frame_offset * sizeof(float),
        sizeof(float),
        NULL
    );

    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.rot,
        &old_rot,
        deshake_ctx->abs_motion.curr_frame_offset * sizeof(float),
        sizeof(float),
        NULL
    );

    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.scale_x,
        &old_scale_x,
        deshake_ctx->abs_motion.curr_frame_offset * sizeof(float),
        sizeof(float),
        NULL
    );

    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.scale_y,
        &old_scale_y,
        deshake_ctx->abs_motion.curr_frame_offset * sizeof(float),
        sizeof(float),
        NULL
    );

    printf("Frame %d:\n", deshake_ctx->curr_frame);

    // printf("    x ");
    new_x = smooth(
        deshake_ctx,
        deshake_ctx->gauss_kernel,
        deshake_ctx->smooth_window,
        input_frame->width,
        deshake_ctx->abs_motion.x
    );
    // printf("    y ");
    new_y = smooth(
        deshake_ctx,
        deshake_ctx->gauss_kernel,
        deshake_ctx->smooth_window,
        input_frame->height,
        deshake_ctx->abs_motion.y
    );
    // printf("    rot ");
    new_rot = smooth(
        deshake_ctx,
        deshake_ctx->gauss_kernel,
        deshake_ctx->smooth_window,
        M_PI / 4,
        deshake_ctx->abs_motion.rot
    );
    // printf("    scale_x ");
    new_scale_x = smooth(
        deshake_ctx,
        deshake_ctx->gauss_kernel,
        deshake_ctx->smooth_window,
        2.0f,
        deshake_ctx->abs_motion.scale_x
    );
    // printf("    scale_y ");
    new_scale_y = smooth(
        deshake_ctx,
        deshake_ctx->gauss_kernel,
        deshake_ctx->smooth_window,
        2.0f,
        deshake_ctx->abs_motion.scale_y
    );

    // printf("Frame %d:\n", deshake_ctx->curr_frame);
    printf("    old_x: %f, old_y: %f\n", old_x, old_y);
    printf("    new_x: %f, new_y: %f\n", new_x, new_y);
    printf("    old_rot: %f, new_rot: %f\n", old_rot, new_rot);
    // printf("    old_scale_x: %f, new_scale_x: %f\n", old_scale_x, new_scale_x);
    // printf("    old_scale_y: %f, new_scale_y: %f\n", old_scale_y, new_scale_y);
    printf("    moving frame %f x, %f y\n", old_x - new_x, old_y - new_y);
    // printf("    rotating %f\n", old_rot - new_rot);

    // TODO: scaling is relative to top-left corner
    // may need to fix that
    affine_transform_matrix(
        old_x - new_x,
        old_y - new_y,
        old_rot - new_rot,
        new_scale_x / old_scale_x,
        new_scale_y / old_scale_y,
        transform
    );

    cle = clEnqueueWriteBuffer(
        deshake_ctx->command_queue,
        deshake_ctx->matrix,
        CL_TRUE,
        0,
        6 * sizeof(float),
        transform,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to write transform to device: %d.\n", cle);

    // TODO: deal with rgb vs yuv input
    src = (cl_mem)input_frame->data[0];
    output_frame = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output_frame) {
        err = AVERROR(ENOMEM);
        goto fail;
    }
    dst = (cl_mem)output_frame->data[0];

    transformed_frame = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!transformed_frame) {
        err = AVERROR(ENOMEM);
        goto fail;
    }
    transformed = (cl_mem)transformed_frame->data[0];

    err = ff_opencl_filter_work_size_from_image(avctx, global_work, input_frame, 0, 0);
    if (err < 0)
        goto fail;

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_transform, 0, cl_mem, &src);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_transform, 1, cl_mem, &transformed);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_transform, 2, cl_mem, &deshake_ctx->matrix);

    cle = clEnqueueNDRangeKernel(
        deshake_ctx->command_queue,
        deshake_ctx->kernel_transform,
        2,
        NULL,
        global_work,
        NULL,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue transform kernel: %d.\n", cle);

    // Run transform kernel
    cle = clFinish(deshake_ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue w/ transform kernel: %d.\n", cle);

    affine_transform_matrix(
        (old_x - new_x) / 3,
        (old_y - new_y) / 3,
        (old_rot - new_rot) / 3,
        new_scale_x / old_scale_x,
        new_scale_y / old_scale_y,
        transform
    );
    update_needed_crop(deshake_ctx, transform, input_frame->width, input_frame->height);

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_crop_upscale, 0, cl_mem, &transformed);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_crop_upscale, 1, cl_mem, &dst);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_crop_upscale, 2, cl_float2, &deshake_ctx->crop.top_left);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_crop_upscale, 3, cl_float2, &deshake_ctx->crop.bottom_right);

    cle = clEnqueueNDRangeKernel(
        deshake_ctx->command_queue,
        deshake_ctx->kernel_crop_upscale,
        2,
        NULL,
        global_work,
        NULL,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue crop_upscale kernel: %d.\n", cle);

    // Run crop_upscale kernel
    cle = clFinish(deshake_ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue w/ crop_upscale kernel: %d.\n", cle);

    err = av_frame_copy_props(output_frame, input_frame);
    if (err < 0)
        goto fail;

    if (deshake_ctx->curr_frame < deshake_ctx->smooth_window / 2) {
        // This means we are somewhere at the start of the video. We need to
        // increment the current frame offset until it reaches the center of
        // the ringbuffers (as the current frame will be located there for
        // the rest of the video).
        //
        // The end of the video is taken care of by draining motion data
        // one-by-one out of the buffer, causing the (at that point fixed)
        // offset to move towards later frames' data.
        ++deshake_ctx->abs_motion.curr_frame_offset;
    }

    ++deshake_ctx->curr_frame;
    av_frame_free(&input_frame);
    av_frame_free(&transformed_frame);
    return ff_filter_frame(outlink, output_frame);

fail:
    clFinish(deshake_ctx->command_queue);
    av_frame_free(&input_frame);
    av_frame_free(&transformed_frame);
    av_frame_free(&output_frame);
    return err;
}

// Add the given frame to the frame queue to eventually be processed.
//
// Also determines the motion from the previous frame and updates the stored
// motion information accordingly.
static int queue_frame(AVFilterLink *link, AVFrame *input_frame) {
    AVFilterContext *avctx = link->dst;
    DeshakeOpenCLContext *deshake_ctx = avctx->priv;
    int err;
    int num_vectors;
    cl_int cle;
    FrameMotion relative;
    SimilarityMatrix model;
    size_t global_work[2];
    cl_mem src, temp;
    float x_trans, y_trans, rot, scale_x, scale_y;

    const int harris_radius = HARRIS_RADIUS;
    const int match_search_radius = MATCH_SEARCH_RADIUS;

    // TODO: deal with rgb vs yuv input
    src = (cl_mem)input_frame->data[0];

    // TODO: Set up kernels so clFinish only gets called once

    err = ff_opencl_filter_work_size_from_image(avctx, global_work, input_frame, 0, 0);
    if (err < 0)
        goto fail;

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_harris, 0, cl_mem, &src);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_harris, 1, cl_mem, &deshake_ctx->harris_buf);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_harris, 2, int, &harris_radius);

    cle = clEnqueueNDRangeKernel(
        deshake_ctx->command_queue,
        deshake_ctx->kernel_harris,
        2,
        NULL,
        global_work,
        NULL,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue harris kernel: %d.\n", cle);

    // Run harris kernel
    cle = clFinish(deshake_ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue w/ harris kernel: %d.\n", cle);

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_nonmax_suppress, 0, cl_mem, &deshake_ctx->harris_buf);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_nonmax_suppress, 1, cl_mem, &deshake_ctx->harris_buf_suppressed);

    cle = clEnqueueNDRangeKernel(
        deshake_ctx->command_queue,
        deshake_ctx->kernel_nonmax_suppress,
        2,
        NULL,
        global_work,
        NULL,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue nonmax_suppress kernel: %d.\n", cle);

    // Run nonmax_suppress kernel
    cle = clFinish(deshake_ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue w/ nonmax_suppress kernel: %d.\n", cle);

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_brief_descriptors, 0, cl_mem, &src);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_brief_descriptors, 1, cl_mem, &deshake_ctx->harris_buf_suppressed);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_brief_descriptors, 2, cl_mem, &deshake_ctx->descriptors);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_brief_descriptors, 3, cl_mem, &deshake_ctx->brief_pattern);

    cle = clEnqueueNDRangeKernel(
        deshake_ctx->command_queue,
        deshake_ctx->kernel_brief_descriptors,
        2,
        NULL,
        global_work,
        NULL,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue brief_descriptors kernel: %d.\n", cle);

    // Run BRIEF kernel
    cle = clFinish(deshake_ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue w/ brief_descriptors kernel: %d.\n", cle);

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 0, cl_mem, &deshake_ctx->descriptors);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 1, cl_mem, &deshake_ctx->prev_descriptors);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 2, cl_mem, &deshake_ctx->matches);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 3, int, &match_search_radius);

    cle = clEnqueueNDRangeKernel(
        deshake_ctx->command_queue,
        deshake_ctx->kernel_match_descriptors,
        2,
        NULL,
        global_work,
        NULL,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue match_descriptors kernel: %d.\n", cle);


    if (av_fifo_size(deshake_ctx->abs_motion.x) == 0) {
        // This is the first frame we've been given to queue, meaning there is
        // no previous frame to match descriptors to

        x_trans = 0.0f;
        y_trans = 0.0f;
        rot = 0.0f;
        scale_x = 1.0f;
        scale_y = 1.0f;

        goto end;
    }

    // Run match_descriptors kernel
    cle = clFinish(deshake_ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue w/ match_descriptors kernel: %d.\n", cle);

    cle = clEnqueueReadBuffer(
        deshake_ctx->command_queue,
        deshake_ctx->matches,
        CL_TRUE,
        0,
        input_frame->width * input_frame->height * sizeof(Vector),
        deshake_ctx->matches_host,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to read matches to host: %d.\n", cle);

    num_vectors = make_vectors_contig(deshake_ctx, input_frame->height, input_frame->width);

    estimate_affine_2d(
        deshake_ctx->matches_contig_host,
        num_vectors,
        model.matrix,
        1.2,
        3000,
        0.999999999
    );

    relative = decompose_transform(model.matrix);

    // Get the absolute transform data for the previous frame
    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.x,
        &x_trans,
        av_fifo_size(deshake_ctx->abs_motion.x) - sizeof(float),
        sizeof(float),
        NULL
    );

    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.y,
        &y_trans,
        av_fifo_size(deshake_ctx->abs_motion.y) - sizeof(float),
        sizeof(float),
        NULL
    );

    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.rot,
        &rot,
        av_fifo_size(deshake_ctx->abs_motion.rot) - sizeof(float),
        sizeof(float),
        NULL
    );

    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.scale_x,
        &scale_x,
        av_fifo_size(deshake_ctx->abs_motion.scale_x) - sizeof(float),
        sizeof(float),
        NULL
    );

    av_fifo_generic_peek_at(
        deshake_ctx->abs_motion.scale_y,
        &scale_y,
        av_fifo_size(deshake_ctx->abs_motion.scale_y) - sizeof(float),
        sizeof(float),
        NULL
    );

    x_trans += relative.translation.s[0];
    y_trans += relative.translation.s[1];
    rot += relative.rotation;
    scale_x /= relative.scale.s[0];
    scale_y /= relative.scale.s[1];
    goto end;

end:
    // Swap the descriptor buffers (we don't need the previous frame's descriptors
    // again so we will use that space for the next frame's descriptors)
    temp = deshake_ctx->prev_descriptors;
    deshake_ctx->prev_descriptors = deshake_ctx->descriptors;
    deshake_ctx->descriptors = temp;

    av_fifo_generic_write(
        deshake_ctx->abs_motion.x,
        &x_trans,
        sizeof(float),
        NULL
    );

    av_fifo_generic_write(
        deshake_ctx->abs_motion.y,
        &y_trans,
        sizeof(float),
        NULL
    );

    av_fifo_generic_write(
        deshake_ctx->abs_motion.rot,
        &rot,
        sizeof(float),
        NULL
    );

    av_fifo_generic_write(
        deshake_ctx->abs_motion.scale_x,
        &scale_x,
        sizeof(float),
        NULL
    );

    av_fifo_generic_write(
        deshake_ctx->abs_motion.scale_y,
        &scale_y,
        sizeof(float),
        NULL
    );

    return ff_framequeue_add(&deshake_ctx->fq, input_frame);

fail:
    clFinish(deshake_ctx->command_queue);
    av_frame_free(&input_frame);
    return err;
}

static int activate(AVFilterContext *ctx) {
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    DeshakeOpenCLContext *deshake_ctx = ctx->priv;
    AVFrame *frame = NULL;
    int ret, status;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (!deshake_ctx->eof) {
        ret = ff_inlink_consume_frame(inlink, &frame);
        if (ret < 0)
            return ret;
        if (ret > 0) {
            if (!frame->hw_frames_ctx)
                return AVERROR(EINVAL);

            if (!deshake_ctx->initialized) {
                ret = deshake_opencl_init(ctx, frame->width, frame->height);
                if (ret < 0)
                    return ret;
            }

            // If there is no more space in the ringbuffers, remove the oldest
            // values to make room for the new ones
            if (av_fifo_space(deshake_ctx->abs_motion.x) == 0) {
                av_fifo_drain(deshake_ctx->abs_motion.x, sizeof(float));
                av_fifo_drain(deshake_ctx->abs_motion.y, sizeof(float));
                av_fifo_drain(deshake_ctx->abs_motion.rot, sizeof(float));
                av_fifo_drain(deshake_ctx->abs_motion.scale_x, sizeof(float));
                av_fifo_drain(deshake_ctx->abs_motion.scale_y, sizeof(float));
            }
            ret = queue_frame(inlink, frame);
            if (ret < 0)
                return ret;
            if (ret >= 0) {
                // See if we have enough buffered frames to process one
                //
                // "enough" is half the smooth window of queued frames into the future
                if (ff_framequeue_queued_frames(&deshake_ctx->fq) >= deshake_ctx->smooth_window / 2) {
                    return filter_frame(inlink, ff_framequeue_take(&deshake_ctx->fq));
                }
            }
        }
    }

    if (!deshake_ctx->eof && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            deshake_ctx->eof = true;
        }
    }

    if (deshake_ctx->eof) {
        // Finish processing the rest of the frames in the queue.
        while(ff_framequeue_queued_frames(&deshake_ctx->fq) != 0) {
            av_fifo_drain(deshake_ctx->abs_motion.x, sizeof(float));
            av_fifo_drain(deshake_ctx->abs_motion.y, sizeof(float));
            av_fifo_drain(deshake_ctx->abs_motion.rot, sizeof(float));
            av_fifo_drain(deshake_ctx->abs_motion.scale_x, sizeof(float));
            av_fifo_drain(deshake_ctx->abs_motion.scale_y, sizeof(float));

            ret = filter_frame(inlink, ff_framequeue_take(&deshake_ctx->fq));
            if (ret < 0) {
                return ret;
            }
        }

        ff_outlink_set_status(outlink, AVERROR_EOF, deshake_ctx->duration);
        return 0;
    }

    if (!deshake_ctx->eof) {
        FF_FILTER_FORWARD_WANTED(outlink, inlink);
    }

    return FFERROR_NOT_READY;
}

static av_cold void deshake_opencl_uninit(AVFilterContext *avctx)
{
    DeshakeOpenCLContext *ctx = avctx->priv;
    cl_int cle;

    uninit_abs_motion_ringbuffers(ctx);

    if (ctx->gauss_kernel)
        av_free(ctx->gauss_kernel);

    if (ctx->small_gauss_kernel)
        av_free(ctx->small_gauss_kernel);

    if (ctx->matches_host)
        av_free(ctx->matches_host);

    if (ctx->matches_contig_host)
        av_free(ctx->matches_contig_host);

    ff_framequeue_free(&ctx->fq);

    if (ctx->kernel_harris) {
        cle = clReleaseKernel(ctx->kernel_harris);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->kernel_nonmax_suppress) {
        cle = clReleaseKernel(ctx->kernel_nonmax_suppress);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->kernel_brief_descriptors) {
        cle = clReleaseKernel(ctx->kernel_brief_descriptors);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->kernel_match_descriptors) {
        cle = clReleaseKernel(ctx->kernel_match_descriptors);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->kernel_crop_upscale) {
        cle = clReleaseKernel(ctx->kernel_crop_upscale);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->kernel_debug_matches) {
        cle = clReleaseKernel(ctx->kernel_debug_matches);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->command_queue) {
        cle = clReleaseCommandQueue(ctx->command_queue);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "command queue: %d.\n", cle);
    }

    if (ctx->harris_buf)
        clReleaseMemObject(ctx->harris_buf);
    if (ctx->harris_buf_suppressed)
        clReleaseMemObject(ctx->harris_buf_suppressed);
    if (ctx->brief_pattern)
        clReleaseMemObject(ctx->brief_pattern);
    if (ctx->descriptors)
        clReleaseMemObject(ctx->descriptors);
    if (ctx->prev_descriptors)
        clReleaseMemObject(ctx->prev_descriptors);
    if (ctx->matches)
        clReleaseMemObject(ctx->matches);
    if (ctx->matches_contig)
        clReleaseMemObject(ctx->matches_contig);
    if (ctx->matrix)
        clReleaseMemObject(ctx->matrix);

    ff_opencl_filter_uninit(avctx);
}

static const AVFilterPad deshake_opencl_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_input,
    },
    { NULL }
};

static const AVFilterPad deshake_opencl_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_output,
    },
    { NULL }
};

#define OFFSET(x) offsetof(DeshakeOpenCLContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption deshake_opencl_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(deshake_opencl);

AVFilter ff_vf_deshake_opencl = {
    .name           = "deshake_opencl",
    // TODO: this
    .description    = NULL_IF_CONFIG_SMALL(""),
    .priv_size      = sizeof(DeshakeOpenCLContext),
    .priv_class     = &deshake_opencl_class,
    .init           = &ff_opencl_filter_init,
    .uninit         = &deshake_opencl_uninit,
    .query_formats  = &ff_opencl_filter_query_formats,
    .activate       = activate,
    .inputs         = deshake_opencl_inputs,
    .outputs        = deshake_opencl_outputs,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE
};
