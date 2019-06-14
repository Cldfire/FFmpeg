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


// Read from a 1d array representing a value for each pixel of a frame given
// the necessary details
static Vector read_from_1d_arrvec(const Vector *buf, int width, int x, int y) {
    return buf[x + y * width];
}

// Returns a random uniformly-distributed number in [low, high]
static int rand_in(int low, int high) {
    double rand_val = rand() / (1.0 + RAND_MAX); 
    int range = high - low + 1;
    int rand_scaled = (rand_val * range) + low;

    return rand_scaled;
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

static int cmp_magnitude(const void *a, const void *b)
{
    return FFDIFFSIGN(((const Vector *)a)->magnitude, ((const Vector *)b)->magnitude);
}

// Cleaned mean (cuts off 20% of values to remove outliers and then averages)
static float clean_mean_magnitude(Vector *vectors, int count)
{
    float mean = 0;
    int cut = count / 5;
    int x;

    AV_QSORT(vectors, count, Vector, cmp_magnitude);

    for (x = cut; x < count - cut; x++) {
        mean += vectors[x].magnitude;
    }

    return mean / (count - cut * 2);
}

// Given a pointer to an array of 3 vectors, replaces the vector closest to the front
// of the array with candidate if:
//   * vectors[i] is an invalid vector (should_consider == false)
//   * vectors[i]'s magnitude and angle are farther from the average than the candidate's
static void update_closest_to_avg(
    Vector *vectors,
    Vector *candidate,
    float avg_magnitude,
    float avg_angle,
    bool ignore_consider
) {
    if (!ignore_consider) {
        if (!candidate->should_consider) {
            return;
        }
    }

    for (int i = 0; i < 3; ++i) {
        Vector *v = &vectors[i];

        if (!v->should_consider) {
            *v = *candidate;
            v->should_consider = true;

            return;
        }
    }

    for (int i = 0; i < 3; ++i) {
        Vector *v = &vectors[i];

        if (
            fabsf(v->magnitude - avg_magnitude) > fabsf(candidate->magnitude - avg_magnitude) &&
            fabsf(v->angle - avg_angle) > fabs(candidate->angle - avg_angle)
         ) {
            *v = *candidate;
            return;
        }
    }
}

// Returns the average angle of the given vectors, dealing with angle wrap-around
// TODO: keep angles in radians to remove need to convert
static float mean_angle(Vector *values, int count)
{
    float sin = 0;
    float cos = 0;

    for (int i = 0; i < count; ++i) {
        float angle = values[i].angle * M_PI / 180.0f;

        sin += sinf(angle);
        cos += cosf(angle);
    }

    return atan2(sin / count, cos / count) * 180.0f / M_PI;
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
    int dx, dy;
    float dx_smooth, dy_smooth;
    float avg_magnitude, avg_angle, angle_diff, magnitude_diff;
    Vector invalid_vector;
    // Holds the three vectors that will be chosen to smooth and base motion off of
    Vector triangle[3];
    // Holds the smoothed points of the triangle from which to transform the current frame
    // as well as the points of the un-smoothed triangle from the previous frame
    SmoothedPointPair triangle_smoothed[3];
    cl_float2 smoothed_point;
    cl_int cle;
    size_t global_work[2];
    size_t global_work_debug[1];
    cl_mem src, dst;

    const int harris_radius = HARRIS_RADIUS;
    const int match_search_radius = MATCH_SEARCH_RADIUS;

    invalid_vector.angle = -1;
    invalid_vector.magnitude = -1;
    invalid_vector.should_consider = false;

    for (int i = 0; i < 3; ++i) {
        triangle[i] = invalid_vector;
    }

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

    avg_magnitude = clean_mean_magnitude(deshake_ctx->matches_contig_host, num_vectors);
    avg_angle = mean_angle(deshake_ctx->matches_contig_host, num_vectors);

    magnitude_diff = avg_magnitude * 0.2;
    angle_diff = 20.0f;

    for (int i = 0; i < num_vectors; ++i) {
        Vector *v = &deshake_ctx->matches_contig_host[i];

        if (fabsf(v->magnitude - avg_magnitude) > magnitude_diff) {
            v->should_consider = false;
            continue;
        }

        // TODO: sometimes (rarely) angles are set up in such a way that this is false for every point
        if (fabsf(v->angle - avg_angle) > angle_diff) {
            v->should_consider = false;
            continue;
        }
    }

    for (int i = 0; i < num_vectors; ++i) {
        Vector *v = &deshake_ctx->matches_contig_host[i];
        update_closest_to_avg(triangle, v, avg_magnitude, avg_angle, false);
    }

    if (num_vectors >= 3) {
        // Make sure we picked 3 points if 3 are available
        // TODO: won't have to do this once we fix angle issue
        for (int i = 0; i < 3; ++i) {
            Vector *t = &triangle[i];

            if (t->should_consider == false) {
                triangle[0] = invalid_vector;
                triangle[1] = invalid_vector;
                triangle[2] = invalid_vector;

                for (int j = 0; j < num_vectors; ++j) {
                    Vector *v = &deshake_ctx->matches_contig_host[j];
                    update_closest_to_avg(triangle, v, avg_magnitude, avg_angle, true);
                }

                goto next;
            }
        }

    next:
        // Generate a one-sided moving exponential average
        dx = triangle[0].p.p2.s[0] - triangle[0].p.p1.s[0];
        dy = triangle[0].p.p2.s[1] - triangle[0].p.p1.s[1];

        deshake_ctx->motion_avg.s[0] = 
            deshake_ctx->alpha * dx + (1.0 - deshake_ctx->alpha) * deshake_ctx->motion_avg.s[0];

        deshake_ctx->motion_avg.s[1] =
            deshake_ctx->alpha * dy + (1.0 - deshake_ctx->alpha) * deshake_ctx->motion_avg.s[1];

        // Remove the average from the current motion to detect the motion that
        // is not on purpose, just as jitter from bumping the camera
        for (int i = 0; i < 3; ++i) {
            Vector *v = &triangle[i];

            dx = v->p.p2.s[0] - v->p.p1.s[0];
            dy = v->p.p2.s[1] - v->p.p1.s[1];

            dx_smooth = dx - deshake_ctx->motion_avg.s[0];
            dy_smooth = dy - deshake_ctx->motion_avg.s[1];

            // Invert the motion to undo it
            // dx_smooth *= -1.0f;
            // dy_smooth *= -1.0f;

            smoothed_point.s[0] = v->p.p1.s[0] + dx_smooth;
            smoothed_point.s[1] = v->p.p1.s[1] + dy_smooth;

            triangle_smoothed[i] = (SmoothedPointPair) {
                v->p.p2,
                smoothed_point
            };
        }
    } else {
        // TODO: do something here
    }


    cle = clEnqueueWriteBuffer(
        deshake_ctx->command_queue,
        deshake_ctx->matches_contig,
        CL_TRUE,
        0,
        3 * sizeof(Vector),
        triangle,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to write vectors buffer to device: %d.\n", cle);

    cle = clEnqueueWriteBuffer(
        deshake_ctx->command_queue,
        deshake_ctx->triangle_smoothed,
        CL_TRUE,
        0,
        3 * sizeof(SmoothedPointPair),
        triangle_smoothed,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to write triangle_smoothed buffer to device: %d.\n", cle);

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_triangle_deshake, 0, cl_mem, &src);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_triangle_deshake, 1, cl_mem, &dst);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_triangle_deshake, 2, cl_mem, &deshake_ctx->triangle_smoothed);

    cle = clEnqueueNDRangeKernel(
        deshake_ctx->command_queue,
        deshake_ctx->kernel_triangle_deshake,
        2,
        NULL,
        global_work,
        NULL,
        0,
        NULL,
        NULL
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue triangle_deshake kernel: %d.\n", cle);

    // Run triangle_deshake kernel
    cle = clFinish(deshake_ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue w/ triangle_deshake kernel: %d.\n", cle);

    CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 0, cl_mem, &dst);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 1, int, &input_frame->width);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 2, int, &input_frame->height);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 3, cl_mem, &deshake_ctx->matches_contig);
    CL_SET_KERNEL_ARG(deshake_ctx->kernel_debug_matches, 4, cl_mem, &deshake_ctx->triangle_smoothed);

    global_work_debug[0] = 3;
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
