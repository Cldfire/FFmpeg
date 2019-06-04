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
#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"

// Number of bits for BRIEF descriptors
// Choices: 128, 256, 512
#define BREIFN 512
// Size of the patch from which a BRIEF descriptor is extracted
// TODO: is this the correct patch size
#define BRIEF_PATCH_SIZE 64
#define BRIEF_PATCH_SIZE_HALF BRIEF_PATCH_SIZE / 2

typedef struct DeshakeOpenCLContext {
    OpenCLFilterContext ocf;
    // Whether or not the above `OpenCLFilterContext` has been initialized
    int initialized;

    cl_command_queue command_queue;
    cl_kernel kernel_harris;
    cl_kernel kernel_nonmax_suppress;
    cl_kernel brief_descriptors;

    cl_mem harris_buf;
    // Harris response after non-maximum suppression
    cl_mem harris_buf_suppressed;

    // BRIEF sampling pattern that is randomly initialized
    cl_mem brief_sampler;

    // Feature point descriptors for the current frame
    cl_mem descriptors;
    // Feature point descriptors for the previous frame
    cl_mem prev_descriptors;
} DeshakeOpenCLContext;

typedef struct PointPair {
    cl_int2 p1;
    cl_int2 p2;
} PointPair;

// Returns a random uniformly-distributed number in [low, high]
static int rand_in(int low, int high) {
    double rand_val = rand() / (1.0 + RAND_MAX); 
    int range = low - high + 1;
    int rand_scaled = (rand_val * range) + low;

    return rand_scaled;
}

static int deshake_opencl_init(AVFilterContext *avctx, int frame_width, int frame_height)
{
    DeshakeOpenCLContext *ctx = avctx->priv;
    // Pointer to the host-side sampler buffer to be initialized and then copied
    // to the GPU
    PointPair *sampler_host;
    cl_int cle;
    int err;

    const int harris_buf_size = frame_height * frame_width * sizeof(float);
    const int descriptor_buf_size = frame_height * frame_width * (BREIFN / 8);

    sampler_host = av_malloc_array(BREIFN, sizeof(PointPair));
    if (!sampler_host)
        return AVERROR(ENOMEM);

    for (int i = 0; i < BREIFN; ++i) {
        PointPair pair;
        
        for (int j = 0; j < 2; ++j) {
            pair.p1.s[j] = rand_in(-BRIEF_PATCH_SIZE_HALF, BRIEF_PATCH_SIZE_HALF);
            pair.p2.s[j] = rand_in(-BRIEF_PATCH_SIZE_HALF, BRIEF_PATCH_SIZE_HALF);
        }

        sampler_host[i] = pair;
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

    ctx->brief_descriptors = clCreateKernel(ctx->ocf.program, "brief_descriptors", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create brief_descriptors kernel: %d.\n", cle);

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

    ctx->brief_sampler = clCreateBuffer(
        ctx->ocf.hwctx->context,
        0,
        sizeof(PointPair) * BREIFN,
        sampler_host,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create brief_sampler buffer: %d.\n", cle);

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

    ctx->initialized = 1;
    av_free(sampler_host);

    return 0;

fail:
    if (sampler_host)
        av_free(sampler_host);
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel_harris)
        clReleaseKernel(ctx->kernel_harris);
    if (ctx->kernel_nonmax_suppress)
        clReleaseKernel(ctx->kernel_nonmax_suppress);
    if (ctx->brief_descriptors)
        clReleaseKernel(ctx->brief_descriptors);
    if (ctx->harris_buf)
        clReleaseMemObject(ctx->harris_buf);
    if (ctx->harris_buf_suppressed)
        clReleaseMemObject(ctx->harris_buf_suppressed);
    if (ctx->brief_sampler)
        clReleaseMemObject(ctx->brief_sampler);
    if (ctx->descriptors)
        clReleaseMemObject(ctx->descriptors);
    if (ctx->prev_descriptors)
        clReleaseMemObject(ctx->prev_descriptors);
    return err;
}

static int filter_frame(AVFilterLink *link, AVFrame *input_frame)
{
    AVFilterContext *avctx = link->dst;
    AVFilterLink *outlink = avctx->outputs[0];
    DeshakeOpenCLContext *deshake_ctx = avctx->priv;
    AVFrame *output_frame = NULL;
    int err;
    cl_int cle;
    size_t global_work[2];
    cl_mem src, dst;

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

    CL_SET_KERNEL_ARG(deshake_ctx->brief_descriptors, 0, cl_mem, &src);
    CL_SET_KERNEL_ARG(deshake_ctx->brief_descriptors, 1, cl_mem, &deshake_ctx->harris_buf_suppressed);
    CL_SET_KERNEL_ARG(deshake_ctx->brief_descriptors, 2, cl_mem, &deshake_ctx->descriptors);
    CL_SET_KERNEL_ARG(deshake_ctx->brief_descriptors, 3, cl_mem, &deshake_ctx->brief_sampler);

    cle = clEnqueueNDRangeKernel(
        deshake_ctx->command_queue,
        deshake_ctx->brief_descriptors,
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

    err = av_frame_copy_props(output_frame, input_frame);
    if (err < 0)
        goto fail;

    av_frame_free(&input_frame);

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

    if (ctx->brief_descriptors) {
        cle = clReleaseKernel(ctx->brief_descriptors);
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
    if (ctx->brief_sampler)
        clReleaseMemObject(ctx->brief_sampler);
    if (ctx->descriptors)
        clReleaseMemObject(ctx->descriptors);
    if (ctx->prev_descriptors)
        clReleaseMemObject(ctx->prev_descriptors);

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
