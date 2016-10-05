#include <stdint.h>
#include <stdbool.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>

#define READ_FRAME_SUCCESS 0
#define READ_FRAME_ERROR (-1)

static inline bool
is_video_stream(AVStream *av_stream)
{
	return (av_stream->codec->codec_type == AVMEDIA_TYPE_VIDEO);
}

/**
 * Opens up format_context, finds and prints AV stream info for filename.
 *
 * @param format_context AV format context to be opened.
 * @param in_filename    Filename to read AV streams from.
 *
 * @note On success, must call avformat_close_input on format_context.
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
 * video codec contex.
 *
 * @param format_context AV format where video streams should be searched for.
 *
 * @return Codec context for first video stream in format_contex on success,
 * NULL on failure.
 */
static AVCodecContext *
find_video_codec_context(AVFormatContext *format_context)
{
	AVStream *video_stream;
	uint32_t stream_index;
	for (stream_index = 0;
	     stream_index < format_context->nb_streams;
	     ++stream_index) {
		video_stream = format_context->streams[stream_index];

		if (is_video_stream(video_stream))
			break;
	}

	if (stream_index >= format_context->nb_streams)
		return NULL;

	return video_stream->codec;
}

/**
 * Allocates a copy of codec_context, and opens the copy. The copy is needed
 * because we cannot call avcodec_open2 on an av_stream's codec context
 * directly.
 *
 * @param codec_context An AV stream's codec context.
 *
 * @note If successful, codec_ctx_copy must be freed with avcodec_free_context,
 * and closed with avcodec_close.
 *
 * @return Opened copy of codec_context on success, NULL on failure.
 */
AVCodecContext *
open_codec_ctx_copy(AVCodecContext *codec_context)
{
	int32_t status;
	AVCodecContext *codec_ctx_copy;
	AVCodec *video_codec;

	video_codec = avcodec_find_decoder(codec_context->codec_id);
	if (video_codec == NULL)
		return NULL;

	codec_ctx_copy = avcodec_alloc_context3(video_codec);
	if (codec_ctx_copy == NULL)
		return NULL;

	status = avcodec_copy_context(codec_ctx_copy, codec_context);
	if (status != 0) {
		avcodec_free_context(&codec_ctx_copy);
		return NULL;
	}

	status = avcodec_open2(codec_ctx_copy, video_codec, NULL);
	if (status != 0) {
		avcodec_free_context(&codec_ctx_copy);
		return NULL;
	}

	return codec_ctx_copy;
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

	AVCodecContext *codec_context = find_video_codec_context(format_context);
	if (codec_context == NULL) {
		status = EXIT_FAILURE;
		goto clean_up_input;
	}

	AVCodecContext *codec_ctx_copy = open_codec_ctx_copy(codec_context);
	if (codec_ctx_copy == NULL) {
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

	frame_rgb->format = codec_ctx_copy->pix_fmt;
	frame_rgb->width = codec_ctx_copy->width;
	frame_rgb->height = codec_ctx_copy->height;

	status = av_image_alloc(frame_rgb->data,
				frame->linesize,
				codec_ctx_copy->width,
				codec_ctx_copy->height,
				codec_ctx_copy->pix_fmt,
				32);
	if (status < 0)
		goto clean_up_frame_rgb;

	av_freep(frame_rgb->data);
clean_up_frame_rgb:
	av_frame_free(&frame_rgb);
clean_up_frame:
	av_frame_free(&frame);
clean_up_codec_ctx_copy:
	avcodec_close(codec_ctx_copy);
	avcodec_free_context(&codec_ctx_copy);
clean_up_input:
	avformat_close_input(&format_context);

	return status;
}
