#include "opencv2/opencv.hpp"
#include <memory>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

int main(int argc, char **argv)
{
        if (argc < 2) {
                printf("Usage: %s path_to_video\n", argv[0]);
                return EXIT_FAILURE;
        }

        cv::VideoCapture video_capture;
        video_capture.open(std::string{argv[1]});
        if (!video_capture.isOpened())
                return EXIT_FAILURE;

        const std::string window_name = "example1";
        cv::namedWindow(window_name);

        cv::Mat frame;
        for (;;) {
                video_capture >> frame;
                if (frame.data == nullptr)
                        break;

                cv::imshow(window_name, frame);
                cv::waitKey(0);
        }
}
