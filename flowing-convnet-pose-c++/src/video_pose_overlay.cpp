#include "estimate_pose.h"
#include "opencv2/opencv.hpp"
#include <memory>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

/*
 * NOTE(brendan): demo code only, which takes a video stream and does
 * frame-by-frame human pose estimation, outputting the resultant video stream
 * in AVI format.
 */
int main(int argc, char **argv)
{
        if (argc < 5) {
                printf("Usage: %s "
                       "<input_path_to_video> "
                       "<output_path_to_video> "
                       "<model> "
                       "<trained_weights>\n",
                       argv[0]);
                return EXIT_FAILURE;
        }

        std::unique_ptr<caffe::Net<float>> heatmap_net =
                init_pose_estimator_network(std::string{argv[3]},
                                            std::string{argv[4]});

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
        for (uint32_t frame_count = 0;
             ;
             ++frame_count) {
                video_capture >> frame;
                if (frame.data == nullptr)
                        break;

                image_pose_overlay(*heatmap_net, frame);

                video_writer << frame;
        }

        video_writer.release();
        video_capture.release();
}
