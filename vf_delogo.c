/*
 * Copyright (c) 2002 Jindrich Makovicka <makovick@gmail.com>
 * Copyright (c) 2011 Stefano Sabatini
 * Copyright (c) 2013, 2015 Jean Delvare <jdelvare@suse.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/**
 * @file
 * A very simple tv station logo remover
 * Originally imported from MPlayer libmpcodecs/vf_delogo.c,
 * the algorithm was later improved.
 */

#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

/**
 * Apply a simple delogo algorithm to the image in src and put the
 * result in dst.
 *
 * The algorithm is only applied to the region specified by the logo
 * parameters.
 *
 * @param w      width of the input image
 * @param h      height of the input image
 * @param logo_x x coordinate of the top left corner of the logo region
 * @param logo_y y coordinate of the top left corner of the logo region
 * @param logo_w width of the logo
 * @param logo_h height of the logo
 * @param band   the size of the band around the processed area
 * @param show   show a rectangle around the processed area, useful for
 *               parameters tweaking
 * @param direct if non-zero perform in-place processing
 */
static void apply_delogo(uint8_t *dst, int dst_linesize,
                         uint8_t *src, int src_linesize,
                         int w, int h, AVRational sar,
                         int logo_x, int logo_y, int logo_w, int logo_h,
                         unsigned int band, double *uglarmtable,
                         double *uglarmweightsum, int show, int direct)
{
    int x, y;
    uint64_t interp, weightl, weightr, weightt, weightb, weight;
    uint8_t *xdst, *xsrc;

    uint8_t *topleft, *botleft, *topright;
    unsigned int left_sample, right_sample;
    int xclipl, xclipr, yclipt, yclipb;
    int logo_x1, logo_x2, logo_y1, logo_y2;

    xclipl = FFMAX(-logo_x, 0);
    xclipr = FFMAX(logo_x+logo_w-w, 0);
    yclipt = FFMAX(-logo_y, 0);
    yclipb = FFMAX(logo_y+logo_h-h, 0);

    logo_x1 = logo_x + xclipl;
    logo_x2 = logo_x + logo_w - xclipr - 1;
    logo_y1 = logo_y + yclipt;
    logo_y2 = logo_y + logo_h - yclipb - 1;

    topleft  = src+logo_y1 * src_linesize+logo_x1;
    topright = src+logo_y1 * src_linesize+logo_x2;
    botleft  = src+logo_y2 * src_linesize+logo_x1;

    if (!direct)
        av_image_copy_plane(dst, dst_linesize, src, src_linesize, w, h);

    dst += (logo_y1 + 1) * dst_linesize;
    src += (logo_y1 + 1) * src_linesize;

	if (!uglarmtable) {
    for (y = logo_y1+1; y < logo_y2; y++) {
        left_sample = topleft[src_linesize*(y-logo_y1)]   +
                      topleft[src_linesize*(y-logo_y1-1)] +
                      topleft[src_linesize*(y-logo_y1+1)];
        right_sample = topright[src_linesize*(y-logo_y1)]   +
                       topright[src_linesize*(y-logo_y1-1)] +
                       topright[src_linesize*(y-logo_y1+1)];

        for (x = logo_x1+1,
             xdst = dst+logo_x1+1,
             xsrc = src+logo_x1+1; x < logo_x2; x++, xdst++, xsrc++) {

            if (show && (y == logo_y1+1 || y == logo_y2-1 ||
                         x == logo_x1+1 || x == logo_x2-1)) {
                *xdst = 0;
                continue;
            }

            /* Weighted interpolation based on relative distances, taking SAR into account */
            weightl = (uint64_t)              (logo_x2-x) * (y-logo_y1) * (logo_y2-y) * sar.den;
            weightr = (uint64_t)(x-logo_x1)               * (y-logo_y1) * (logo_y2-y) * sar.den;
            weightt = (uint64_t)(x-logo_x1) * (logo_x2-x)               * (logo_y2-y) * sar.num;
            weightb = (uint64_t)(x-logo_x1) * (logo_x2-x) * (y-logo_y1)               * sar.num;

            interp =
                left_sample * weightl
                +
                right_sample * weightr
                +
                (topleft[x-logo_x1]    +
                 topleft[x-logo_x1-1]  +
                 topleft[x-logo_x1+1]) * weightt
                +
                (botleft[x-logo_x1]    +
                 botleft[x-logo_x1-1]  +
                 botleft[x-logo_x1+1]) * weightb;
            weight = (weightl + weightr + weightt + weightb) * 3U;
            interp = ROUNDED_DIV(interp, weight);

            if (y >= logo_y+band && y < logo_y+logo_h-band &&
                x >= logo_x+band && x < logo_x+logo_w-band) {
                *xdst = interp;
            } else {
                unsigned dist = 0;

                if      (x < logo_x+band)
                    dist = FFMAX(dist, logo_x-x+band);
                else if (x >= logo_x+logo_w-band)
                    dist = FFMAX(dist, x-(logo_x+logo_w-1-band));

                if      (y < logo_y+band)
                    dist = FFMAX(dist, logo_y-y+band);
                else if (y >= logo_y+logo_h-band)
                    dist = FFMAX(dist, y-(logo_y+logo_h-1-band));

                *xdst = (*xsrc*dist + interp*(band-dist))/band;
            }
        }

        dst += dst_linesize;
        src += src_linesize;
    }
} else {
        int bx, by, table_stride;
        double interpd;
        double *table_t, *table_b, *table_l, *table_r;

        table_stride = logo_w - 1;

        for (y = 1; y < logo_y2 - logo_y1; y++) {
            for (x = 1, xdst = dst + logo_x1 + 1, xsrc = src + logo_x1 + 1;
                x < logo_x2 - logo_x1;
                x++, xdst++, xsrc++) {

                if (show && (y == 1 || y == logo_y2 - logo_y1 - 1 ||
                             x == 1 || x == logo_x2 - logo_x1 - 1)) {
                    *xdst = 0;
                    continue;
                }

                table_t = uglarmtable + x + table_stride * y;
                table_b = uglarmtable + x + table_stride * (logo_h - y - 1);
                table_l = uglarmtable + table_stride * (y - 1) + x;
                table_r = uglarmtable + table_stride * (y - 1) + logo_w - x - 1;

                interpd = 0;

                /* top+bottom on the left of the current point */
                for (bx = 0; bx < x; bx++) {
                    interpd += topleft[bx] * *table_t;
                    interpd += botleft[bx] * *table_b;
                    table_t--;
                    table_b--;
                }
                /* top+bottom on the right of the current point */
                for (; bx < logo_w; bx++) {
                    interpd += topleft[bx] * *table_t;
                    interpd += botleft[bx] * *table_b;
                    table_t++;
                    table_b++;
                }
                /* left+right above the current point */
                for (by = 1; by < y; by++) {
                    interpd += topleft[by * src_linesize] * *table_l;
                    interpd += topleft[by * src_linesize + table_stride] * *table_r;
                    table_l -= table_stride;
                    table_r -= table_stride;
                }
                /* left+right below the current point */
                for (; by < logo_h - 1; by++) {
                    interpd += topleft[by * src_linesize] * *table_l;
                    interpd += topleft[by * src_linesize + table_stride] * *table_r;
                    table_l += table_stride;
                    table_r += table_stride;
                }

                av_assert2(table_t == uglarmtable + (logo_w - x) + table_stride * y);
                av_assert2(table_b == uglarmtable + (logo_w - x) + table_stride * (logo_h - y - 1));
                av_assert2(table_l == uglarmtable + table_stride * (logo_h - y - 1) + x);
                av_assert2(table_r == uglarmtable + table_stride * (logo_h - y - 1) + logo_w - x - 1);

                interp = (uint64_t)(interpd / uglarmweightsum[x - 1 + (y - 1) * (logo_w - 2)]);
					*xdst = interp;
            }

            dst += dst_linesize;
            src += src_linesize;
        }
    }
}

/**
 * Calculate the lookup tables to be used in UGLARM interpolation mode.
 *
 * @param *uglarmtable      Pointer to table containing weights for each possible
 *                          diagonal distance between a border pixel and an inner
 *                          logo pixel.
 * @param *uglarmweightsum  Pointer to a table containing the weight sum to divide
 *                          by for each pixel within the logo area.
 * @param sar               The sar to take into account when calculating lookup
 *                          tables.
 * @param logo_w            width of the logo
 * @param logo_h            height of the logo
 * @param exponent          exponent used in uglarm interpolation
 */
static void calc_uglarm_tables(double *uglarmtable, double *uglarmweightsum,
                               AVRational sar, int logo_w, int logo_h,
                               float exponent)
{
    double aspect = (double)sar.num / sar.den;
    double aspect2 = aspect * aspect;
    int x, y;

    /* uglarmtable will contain a weight for each possible diagonal distance
     * between a border pixel and an inner logo pixel. The maximum distance in
     * each direction between border and an inner pixel can be logo_w - 1. The
     * weight of a border pixel which is x,y pixels away is stored at position
     * x + y * (logo_w - 1). */
    for (y = 0; y < logo_h - 1; y++)
        for (x = 0; x < logo_w - 1; x++) {
            if (x + y != 0) {
                double d = pow(x * x * aspect2 + y * y, exponent / 2);
                uglarmtable[x + y * (logo_w - 1)] = 1.0 / d;
            } else {
                uglarmtable[x + y * (logo_w - 1)] = 1.0;
            }
        }

    /* uglarmweightsum will contain the sum of all weights which is used when
     * an inner pixel of the logo at position x,y is calculated out of the
     * border pixels. The aggregated value has to be divided by that. The value
     * to use for the inner 1-based logo position x,y is stored at
     * (x - 1) + (y - 1) * (logo_w - 2). */
    for (y = 1; y < logo_h - 1; y++)
        for (x = 1; x < logo_w - 1; x++) {
            double weightsum = 0;

            for (int bx = 0; bx < logo_w; bx++) {
                /* top border */
                weightsum += uglarmtable[abs(bx - x) + y * (logo_w - 1)];
                /* bottom border */
                weightsum += uglarmtable[abs(bx - x) + (logo_h - y - 1) * (logo_w - 1)];
            }

            for (int by = 1; by < logo_h - 1; by++) {
                /* left border */
                weightsum += uglarmtable[x + abs(by - y) * (logo_w - 1)];
                /* right border */
                weightsum += uglarmtable[(logo_w - x - 1) + abs(by - y) * (logo_w - 1)];
            }

            uglarmweightsum[(x - 1) + (y - 1) * (logo_w - 2)] = weightsum;
        }
 }
 
enum mode {
    MODE_XY,
    MODE_UGLARM,
    MODE_NB
};

#define MAX_SUB 2

typedef struct DelogoContext {
    const AVClass *class;
    int x, y, w, h, band, mode, show;
	float exponent;
	double *uglarmtable[MAX_SUB + 1][MAX_SUB + 1], *uglarmweightsum[MAX_SUB + 1][MAX_SUB + 1];
}  DelogoContext;

#define OFFSET(x) offsetof(DelogoContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption delogo_options[]= {
    { "x",    "set logo x position",       OFFSET(x),    AV_OPT_TYPE_INT, { .i64 = -1 }, -1, INT_MAX, FLAGS },
    { "y",    "set logo y position",       OFFSET(y),    AV_OPT_TYPE_INT, { .i64 = -1 }, -1, INT_MAX, FLAGS },
    { "w",    "set logo width",            OFFSET(w),    AV_OPT_TYPE_INT, { .i64 = -1 }, -1, INT_MAX, FLAGS },
    { "h",    "set logo height",           OFFSET(h),    AV_OPT_TYPE_INT, { .i64 = -1 }, -1, INT_MAX, FLAGS },
#if LIBAVFILTER_VERSION_MAJOR < 7
    /* Actual default value for band/t is 1, set in init */
    { "band", "set delogo area band size", OFFSET(band), AV_OPT_TYPE_INT, { .i64 =  0 },  0, INT_MAX, FLAGS },
    { "t",    "set delogo area band size", OFFSET(band), AV_OPT_TYPE_INT, { .i64 =  0 },  0, INT_MAX, FLAGS },
#endif
    { "mode", "set the interpolation mode",OFFSET(mode), AV_OPT_TYPE_INT, { .i64 = MODE_XY }, MODE_XY, MODE_NB-1, FLAGS, "mode" },
        { "xy",     "use pixels in straight x and y direction",  OFFSET(mode), AV_OPT_TYPE_CONST, { .i64 = MODE_XY },     0, 0, FLAGS, "mode" },
        { "uglarm", "UGLARM mode, use full border",              OFFSET(mode), AV_OPT_TYPE_CONST, { .i64 = MODE_UGLARM }, 0, 0, FLAGS, "mode" },
    { "exponent","exponent used for UGLARM interpolation", OFFSET(exponent), AV_OPT_TYPE_FLOAT, { .dbl = 3.0 }, 0, 6, FLAGS },
    { "show", "show delogo area",          OFFSET(show), AV_OPT_TYPE_BOOL,{ .i64 =  0 },  0, 1,       FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(delogo);

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV444P,  AV_PIX_FMT_YUV422P,  AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUV411P,  AV_PIX_FMT_YUV410P,  AV_PIX_FMT_YUV440P,
        AV_PIX_FMT_YUVA420P, AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static av_cold int init(AVFilterContext *ctx)
{
    DelogoContext *s = ctx->priv;

#define CHECK_UNSET_OPT(opt)                                            \
    if (s->opt == -1) {                                            \
        av_log(s, AV_LOG_ERROR, "Option %s was not set.\n", #opt); \
        return AVERROR(EINVAL);                                         \
    }
    CHECK_UNSET_OPT(x);
    CHECK_UNSET_OPT(y);
    CHECK_UNSET_OPT(w);
    CHECK_UNSET_OPT(h);

#if LIBAVFILTER_VERSION_MAJOR < 7
    if (s->band == 0) { /* Unset, use default */
        av_log(ctx, AV_LOG_WARNING, "Note: default band value was changed from 4 to 1.\n");
        s->band = 1;
    } else if (s->band != 1) {
        av_log(ctx, AV_LOG_WARNING, "Option band is deprecated.\n");
    }
#else
    s->band = 1;
#endif
    av_log(ctx, AV_LOG_VERBOSE, "x:%d y:%d, w:%d h:%d band:%d mode:%d exponent:%f show:%d\n",
           s->x, s->y, s->w, s->h, s->band, s->mode, s->exponent, s->show);

    s->w += s->band*2;
    s->h += s->band*2;
    s->x -= s->band;
    s->y -= s->band;

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    DelogoContext *s = ctx->priv;

    for (int hsub = 0; hsub <= MAX_SUB; hsub++)
        for (int vsub = 0; vsub <= MAX_SUB; vsub++) {
            av_freep(&s->uglarmtable[hsub][vsub]);
            av_freep(&s->uglarmweightsum[hsub][vsub]);
        }
}

static int config_input(AVFilterLink *inlink)
{
    DelogoContext *s = inlink->dst->priv;

    /* Check whether the logo area fits in the frame */
    if (s->x + (s->band - 1) < 0 || s->x + s->w - (s->band*2 - 2) > inlink->w ||
        s->y + (s->band - 1) < 0 || s->y + s->h - (s->band*2 - 2) > inlink->h) {
        av_log(s, AV_LOG_ERROR, "Logo area is outside of the frame.\n");
        return AVERROR(EINVAL);
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    DelogoContext *s = inlink->dst->priv;
    AVFilterLink *outlink = inlink->dst->outputs[0];
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFrame *out;
    int hsub0 = desc->log2_chroma_w;
    int vsub0 = desc->log2_chroma_h;
    int direct = 0;
    int plane;
    AVRational sar;

    if (av_frame_is_writable(in)) {
        direct = 1;
        out = in;
    } else {
        out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }

        av_frame_copy_props(out, in);
    }

    sar = in->sample_aspect_ratio;
    /* Assume square pixels if SAR is unknown */
    if (!sar.num)
        sar.num = sar.den = 1;

    if (s->mode == MODE_UGLARM)
        av_assert0(hsub0 <= MAX_SUB && vsub0 <= MAX_SUB);

    for (plane = 0; plane < desc->nb_components; plane++) {
        int hsub = plane == 1 || plane == 2 ? hsub0 : 0;
        int vsub = plane == 1 || plane == 2 ? vsub0 : 0;

        /* Up and left borders were rounded down, inject lost bits
         * into width and height to avoid error accumulation */
        int logo_w = AV_CEIL_RSHIFT(s->w + (s->x & ((1<<hsub)-1)), hsub);
        int logo_h = AV_CEIL_RSHIFT(s->h + (s->y & ((1<<vsub)-1)), vsub);

        /* Init lookup tables once */
        if ((s->mode == MODE_UGLARM) && (!s->uglarmtable[hsub][vsub])) {
            s->uglarmtable[hsub][vsub] =
                av_malloc_array((logo_w - 1) * (logo_h - 1), sizeof(*s->uglarmtable[hsub][vsub]));

            if (!s->uglarmtable[hsub][vsub])
                return AVERROR(ENOMEM);

            s->uglarmweightsum[hsub][vsub] =
                av_malloc_array((logo_w - 2) * (logo_h - 2), sizeof(*s->uglarmweightsum[hsub][vsub]));

            if (!s->uglarmweightsum[hsub][vsub])
                return AVERROR(ENOMEM);

            calc_uglarm_tables(s->uglarmtable[hsub][vsub],
                            s->uglarmweightsum[hsub][vsub],
                            sar, logo_w, logo_h, s->exponent);
        }

        apply_delogo(out->data[plane], out->linesize[plane],
                     in ->data[plane], in ->linesize[plane],
                     AV_CEIL_RSHIFT(inlink->w, hsub),
                     AV_CEIL_RSHIFT(inlink->h, vsub),
                     sar, s->x>>hsub, s->y>>vsub,
                     logo_w,
                     logo_h,
                     s->band>>FFMIN(hsub, vsub),
                     s->uglarmtable[hsub][vsub],
                     s->uglarmweightsum[hsub][vsub],
                     s->show, direct);
    }

    if (!direct)
        av_frame_free(&in);

    return ff_filter_frame(outlink, out);
}

static const AVFilterPad avfilter_vf_delogo_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    },
    { NULL }
};

static const AVFilterPad avfilter_vf_delogo_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_delogo = {
    .name          = "delogo",
    .description   = NULL_IF_CONFIG_SMALL("Remove logo from input video."),
    .priv_size     = sizeof(DelogoContext),
    .priv_class    = &delogo_class,
    .init          = init,
	.uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = avfilter_vf_delogo_inputs,
    .outputs       = avfilter_vf_delogo_outputs,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
