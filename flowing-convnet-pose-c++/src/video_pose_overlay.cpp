#include "opencv2/opencv.hpp"
#include <memory>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

int main(int argc, char **argv)
{
        if (argc < 3) {
                printf("Usage: %s input_path_to_video output_path_to_video\n",
                       argv[0]);
                return EXIT_FAILURE;
        }

        cv::VideoCapture video_capture;
        video_capture.open(std::string{argv[1]});
        if (!video_capture.isOpened())
                return EXIT_FAILURE;

        const int32_t frame_width =
                static_cast<int>(video_capture.get(CV_CAP_PROP_FRAME_WIDTH));
        const int32_t frame_height =
                static_cast<int>(video_capture.get(CV_CAP_PROP_FRAME_HEIGHT));

        cv::VideoWriter video_writer;
        video_writer.open(std::string{argv[2]},
                          CV_FOURCC('X', 'V', 'I', 'D'),
                          video_capture.get(CV_CAP_PROP_FPS),
                          cv::Size{frame_width, frame_height});
        if (!video_writer.isOpened())
                return EXIT_FAILURE;

        cv::Mat frame;
        for (;;) {
                video_capture >> frame;
                if (frame.data == nullptr)
                        break;

                video_writer << frame;
        }

        video_writer.release();
        video_capture.release();
}
