#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <cstdio>

int main(int argc, char **argv)
{
        if (argc < 4) {
                printf("Usage: %s image_name model_prototxt model_binaryproto\n",
                       argv[0]);
                return EXIT_FAILURE;
        }

        const std::string image_name{argv[1]};
        const std::string model_prototxt{argv[2]};
        const std::string pretrained_weights{argv[3]};

        cv::Mat image = cv::imread(image_name);
        if (image.empty())
                return EXIT_FAILURE;

        const std::string window_name = "example1";
        cv::namedWindow(window_name);

        cv::imshow(window_name, image);

        cv::waitKey();

        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        caffe::Net<float> heatmap_net{model_prototxt, caffe::TEST};

        heatmap_net.CopyTrainedLayersFrom(pretrained_weights);
        heatmap_net.Forward();
}
