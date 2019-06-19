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
#include "libavutil/qsort.h"
#include "avfilter.h"
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

#define MATCHES_CONTIG_SIZE 500

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

// Stores the translation, scale, rotation, and skew deltas between two frames
typedef struct FrameMotion {
    cl_float2 translation;
    float rotation;
    cl_float2 scale;
    cl_float2 skew;
} FrameMotion;

typedef struct DeshakeOpenCLContext {
    OpenCLFilterContext ocf;
    // Whether or not the above `OpenCLFilterContext` has been initialized
    int initialized;

    // Buffer to copy `matches` into for the CPU to work with
    Vector *matches_host;
    Vector *matches_contig_host;

    // Vector that tracks average x and y motion over time
    // TODO: don't use cl_int2 for this
    cl_float2 motion_avg;
    // Used in motion averaging
    float alpha;

    cl_command_queue command_queue;
    cl_kernel kernel_harris;
    cl_kernel kernel_nonmax_suppress;
    cl_kernel kernel_brief_descriptors;
    cl_kernel kernel_match_descriptors;
    cl_kernel kernel_triangle_deshake;

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
    cl_mem triangle_smoothed;
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
        printf("Max inlier count: %d out of %d matches\n", max_good_count, num_point_pairs);
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

static int deshake_opencl_init(AVFilterContext *avctx, int frame_width, int frame_height)
{
    DeshakeOpenCLContext *ctx = avctx->priv;
    // Pointer to the host-side pattern buffer to be initialized and then copied
    // to the GPU
    PointPair *pattern_host;
    cl_int cle;
    int err;
    cl_ulong8 zeroed_ulong8;

    const int harris_buf_size = frame_height * frame_width * sizeof(float);
    const int descriptor_buf_size = frame_height * frame_width * (BREIFN / 8);

    srand(947247);
    memset(&zeroed_ulong8, 0, sizeof(cl_ulong8));

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

    ctx->motion_avg.s[0] = 0;
    ctx->motion_avg.s[1] = 0;
    // TODO: Make the 20.0f configurable? (Number of frames for averaging window)
    ctx->alpha = 2.0f / 10.0f;

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

    ctx->kernel_triangle_deshake = clCreateKernel(ctx->ocf.program, "triangle_deshake", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel_triangle_deshake kernel: %d.\n", cle);

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

    ctx->triangle_smoothed = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        3 * sizeof(SmoothedPointPair),
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create triangle_smoothed buffer: %d.\n", cle);

    ctx->initialized = 1;
    av_free(pattern_host);

    return 0;

fail:
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
    if (ctx->kernel_triangle_deshake)
        clReleaseKernel(ctx->kernel_triangle_deshake);
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
    if (ctx->triangle_smoothed)
        clReleaseMemObject(ctx->triangle_smoothed);
    return err;
}

static int filter_frame(AVFilterLink *link, AVFrame *input_frame)
{
    AVFilterContext *avctx = link->dst;
    AVFilterLink *outlink = avctx->outputs[0];
    DeshakeOpenCLContext *deshake_ctx = avctx->priv;
    AVFrame *output_frame = NULL;
    int err;
    int num_vectors;
    cl_int cle;
    double model[6];
    FrameMotion motion;
    size_t global_work[2];
    size_t global_work_debug[1];
    cl_mem src, dst;

    static int frame;
    ++frame;

    const int harris_radius = HARRIS_RADIUS;
    const int match_search_radius = MATCH_SEARCH_RADIUS;

    if (!input_frame->hw_frames_ctx)
        return AVERROR(EINVAL);

    if (!deshake_ctx->initialized) {
        err = deshake_opencl_init(avctx, input_frame->width, input_frame->height);
        if (err < 0)
            goto fail;
    }

    // This filter only operates on RGB data and we know that will be on the first plane
    src = (cl_mem)input_frame->data[0];
    output_frame = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output_frame) {
        err = AVERROR(ENOMEM);
        goto fail;
    }
    dst = (cl_mem)output_frame->data[0];

    // TODO: Set up kernels so clFinish only gets called once

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_harris, 0, cl_mem, &src);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_harris, 1, cl_mem, &dst);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_harris, 2, cl_mem, &deshake_ctx->harris_buf);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_harris, 3, int, &harris_radius);

    err = ff_opencl_filter_work_size_from_image(avctx, global_work, input_frame, 0, 0);
    if (err < 0)
        goto fail;

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

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_nonmax_suppress, 0, cl_mem, &src);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_nonmax_suppress, 1, cl_mem, &dst);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_nonmax_suppress, 2, cl_mem, &deshake_ctx->harris_buf);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_nonmax_suppress, 3, cl_mem, &deshake_ctx->harris_buf_suppressed);

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
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_brief_descriptors, 1, cl_mem, &dst);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_brief_descriptors, 2, cl_mem, &deshake_ctx->harris_buf_suppressed);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_brief_descriptors, 3, cl_mem, &deshake_ctx->descriptors);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_brief_descriptors, 4, cl_mem, &deshake_ctx->brief_pattern);

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

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 0, cl_mem, &src);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 1, cl_mem, &dst);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 2, cl_mem, &deshake_ctx->descriptors);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 3, cl_mem, &deshake_ctx->prev_descriptors);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 4, cl_mem, &deshake_ctx->matches);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_match_descriptors, 5, int, &match_search_radius);

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
        model,
        1.5,
        3000,
        0.9999
    );

    motion = decompose_transform(model);

    printf("Frame %d:\n", frame);
    printf("    Translation: x = %f, y = %f\n", motion.translation.s[0], motion.translation.s[1]);
    printf("    Rotation: %f degrees\n", motion.rotation * (180.0 / M_PI));
    printf("    Scale: x = %f, y = %f\n", motion.scale.s[0], motion.scale.s[1]);
    printf("    Skew: x = %f, y = %f\n", motion.skew.s[0], motion.skew.s[1]);

    cle = clEnqueueWriteBuffer(
        deshake_ctx->command_queue,
        deshake_ctx->matches_contig,
        CL_TRUE,
        0,
        num_vectors * sizeof(Vector),
        deshake_ctx->matches_contig_host,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to write vectors buffer to device: %d.\n", cle);

    if (num_vectors != 0) {
        CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 0, cl_mem, &dst);
        CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 1, int, &input_frame->width);
        CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 2, int, &input_frame->height);
        CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 3, cl_mem, &deshake_ctx->matches_contig);

        global_work_debug[0] = num_vectors;
        cle = clEnqueueNDRangeKernel(
            deshake_ctx->command_queue,
            deshake_ctx->kernel_debug_matches,
            1,
            NULL,
            global_work_debug,
            NULL,
            0,
            NULL,
            NULL
        );
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue debug_matches kernel: %d.\n", cle);

        // Run debug_matches kernel
        cle = clFinish(deshake_ctx->command_queue);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue w/ debug_matches kernel: %d.\n", cle);
    }

    err = av_frame_copy_props(output_frame, input_frame);
    if (err < 0)
        goto fail;

    av_frame_free(&input_frame);

    // Swap the descriptor buffers (we don't need the previous frame's descriptors
    // again so we will use that space for the next frame's descriptors)
    cl_mem temp = deshake_ctx->prev_descriptors;
    deshake_ctx->prev_descriptors = deshake_ctx->descriptors;
    deshake_ctx->descriptors = temp;

    return ff_filter_frame(outlink, output_frame);

fail:
    clFinish(deshake_ctx->command_queue);
    av_frame_free(&input_frame);
    av_frame_free(&output_frame);
    return err;
}

static av_cold void deshake_opencl_uninit(AVFilterContext *avctx)
{
    DeshakeOpenCLContext *ctx = avctx->priv;
    cl_int cle;

    if (ctx->matches_host)
        av_free(ctx->matches_host);

    if (ctx->matches_contig_host)
        av_free(ctx->matches_contig_host);

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

    if (ctx->kernel_triangle_deshake) {
        cle = clReleaseKernel(ctx->kernel_triangle_deshake);
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
    if (ctx->triangle_smoothed)
        clReleaseMemObject(ctx->triangle_smoothed);

    ff_opencl_filter_uninit(avctx);
}

static const AVFilterPad deshake_opencl_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
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
    .inputs         = deshake_opencl_inputs,
    .outputs        = deshake_opencl_outputs,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE
};
