#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <cstdio>

typedef struct val_location {
        cv::Point location;
        double val;
} val_location;

static void
channels_from_blob(std::vector<cv::Mat>& input_channels,
                   boost::shared_ptr<caffe::Blob<float>> blob,
                   int32_t width,
                   int32_t height)
{
        float *input_data = blob->mutable_cpu_data();
        for (int32_t channel_index = 0;
             channel_index < blob->channels();
             ++channel_index) {
                cv::Mat channel{height, width, CV_32FC1, input_data};
                input_channels.push_back(channel);
                input_data += width*height;
        }
}

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

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32FC3);

        std::vector<cv::Mat> input_channels;
        channels_from_blob(input_channels,
                           heatmap_net.blobs().front(),
                           image.cols,
                           image.rows);
        cv::split(image, input_channels);

        heatmap_net.Forward();

        auto heatmap_blob = heatmap_net.blob_by_name("conv5_fusion");
        std::vector<cv::Mat> joints_channels;
        channels_from_blob(joints_channels,
                           heatmap_blob,
                           heatmap_blob->shape(3),
                           heatmap_blob->shape(2));

        constexpr uint32_t NUM_JOINTS = 7;
        assert(joints_channels.size() == NUM_JOINTS);

        image.convertTo(image, CV_8UC3);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        val_location joints[NUM_JOINTS];
        for (uint32_t joints_index = 0;
             joints_index < NUM_JOINTS;
             ++joints_index) {
                cv::Mat& joint = joints_channels.at(joints_index);
                cv::Mat joint_resized;
                cv::resize(joint, joint_resized, cv::Size{image.cols, image.rows});

                cv::minMaxLoc(joint_resized,
                              NULL,
                              &joints[joints_index].val,
                              NULL,
                              &joints[joints_index].location);

                cv::circle(image,
                           joints[joints_index].location,
                           5,
                           CV_RGB(0xff, 0, 0));
        }

        cv::imshow(window_name, image);
        cv::waitKey();
}
