#include <stdint.h>
#include <stdbool.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#define READ_FRAME_SUCCESS 0
#define READ_FRAME_ERROR (-1)

/**
 * Opens up format_context, finds and prints AV stream info for filename.
 *
 * @param format_context AV format context to be opened.
 * @param in_filename    Filename to read AV streams from.
 *
 * @warning On success, must call avformat_close_input on format_context.
 *
 * @return READ_FRAME_SUCCESS on success, READ_FRAME_ERROR on failure.
 */
static int32_t
open_av_stream_print_info(AVFormatContext **format_context, char *in_filename)
{
	int32_t status = avformat_open_input(format_context,
					     in_filename,
					     NULL,
					     NULL);
	if (status != 0)
		return status;

	status = avformat_find_stream_info(*format_context, NULL);
	if (status < 0) {
		avformat_close_input(format_context);
		return READ_FRAME_ERROR;
	}

	av_dump_format(*format_context, 0, in_filename, false);

	return READ_FRAME_SUCCESS;
}

/**
 * Finds a video stream for the AV format context and returns the associated
 * stream index.
 *
 * @param format_context AV format where video streams should be searched for.
 *
 * @return Index of video stream on success, negative error code on failure.
 */
static int32_t
find_video_stream_index(AVFormatContext *format_context)
{
	AVStream *video_stream;
	uint32_t stream_index;

	for (stream_index = 0;
	     stream_index < format_context->nb_streams;
	     ++stream_index) {
		video_stream = format_context->streams[stream_index];

		if (video_stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
			break;
	}

	if (stream_index >= format_context->nb_streams)
		return READ_FRAME_ERROR;

	return stream_index;
}

/**
 * Allocates a codec context for video_stream, and opens it.  We cannot call
 * avcodec_open2 on an av_stream's codec context directly.
 *
 * @param video_stream Video stream to open codec context for.
 *
 * @warning If successful, codec_context must be freed with
 * avcodec_free_context, and closed with avcodec_close.
 *
 * @return Opened copy of codec_context on success, NULL on failure.
 */
static AVCodecContext *
open_video_codec_ctx(AVStream *video_stream)
{
	int32_t status;
	AVCodecContext *codec_context;
	AVCodec *video_codec;

	video_codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
	if (video_codec == NULL)
		return NULL;

	codec_context = avcodec_alloc_context3(video_codec);
	if (codec_context == NULL)
		return NULL;

	status = avcodec_parameters_to_context(codec_context,
					       video_stream->codecpar);
	if (status != 0) {
		avcodec_free_context(&codec_context);
		return NULL;
	}

	status = avcodec_open2(codec_context, video_codec, NULL);
	if (status != 0) {
		avcodec_free_context(&codec_context);
		return NULL;
	}

	return codec_context;
}

int main(int32_t argc, char **argv)
{
	int32_t status = EXIT_SUCCESS;
	AVFormatContext *format_context = NULL;

	if (argc < 2)
		return EXIT_FAILURE;

	av_register_all();

	status = open_av_stream_print_info(&format_context, argv[1]);
	if (status != READ_FRAME_SUCCESS)
		return status;

	int32_t video_stream_index = find_video_stream_index(format_context);
	if (video_stream_index < 0) {
		status = video_stream_index;
		goto clean_up_input;
	}

	AVCodecContext *codec_context =
		open_video_codec_ctx(format_context->streams[video_stream_index]);
	if (codec_context == NULL) {
		status = EXIT_FAILURE;
		goto clean_up_input;
	}

	AVFrame *frame = av_frame_alloc();
	if (frame == NULL) {
		status = EXIT_FAILURE;
		goto clean_up_codec_ctx_copy;
	}

	AVFrame *frame_rgb = av_frame_alloc();
	if (frame_rgb == NULL) {
		status = EXIT_FAILURE;
		goto clean_up_frame;
	}

	frame_rgb->format = codec_context->pix_fmt;
	frame_rgb->width = codec_context->width;
	frame_rgb->height = codec_context->height;

	status = av_image_alloc(frame_rgb->data,
				frame_rgb->linesize,
				codec_context->width,
				codec_context->height,
				codec_context->pix_fmt,
				32);
	if (status < 0)
		goto clean_up_frame_rgb;

	struct SwsContext *sws_context = sws_getContext(codec_context->width,
							codec_context->height,
							codec_context->pix_fmt,
							codec_context->width,
							codec_context->height,
							AV_PIX_FMT_RGB24,
							SWS_BILINEAR,
							NULL,
							NULL,
							NULL);
	if (sws_context == NULL) {
		status = EXIT_FAILURE;
		goto clean_up_frame_rgb_data;
	}

	AVPacket packet;
	while (av_read_frame(format_context, &packet) == 0) {
		if (packet.stream_index == video_stream_index) {
			status = avcodec_send_packet(codec_context, &packet);
			if (status != 0) {
				av_packet_unref(&packet);
				goto clean_up_sws_context;
			}

			status = avcodec_receive_frame(codec_context, frame);
			if ((status != 0) && (status != AVERROR(EAGAIN))) {
				av_packet_unref(&packet);
				goto clean_up_sws_context;
			}

			// TODO(brendan): on status != AVERROR(EAGAIN), scale
			// frame to RGB and save to file.

			sws_scale(sws_context,
				  (const uint8_t * const *)frame->data,
				  frame->linesize,
				  0,
				  frame->height,
				  frame_rgb->data,
				  frame_rgb->linesize);

			av_frame_unref(frame);
		}

		av_packet_unref(&packet);
	}

clean_up_sws_context:
	sws_freeContext(sws_context);
clean_up_frame_rgb_data:
	av_freep(frame_rgb->data);
clean_up_frame_rgb:
	av_frame_free(&frame_rgb);
clean_up_frame:
	av_frame_free(&frame);
clean_up_codec_ctx_copy:
	avcodec_close(codec_context);
	avcodec_free_context(&codec_context);
clean_up_input:
	avformat_close_input(&format_context);

	return status;
}
