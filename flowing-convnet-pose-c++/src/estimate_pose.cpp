#include "estimate_pose.h"
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"
#include "opencv2/opencv.hpp"
#include <memory>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

constexpr uint32_t NUM_JOINTS = 7;
constexpr uint32_t NUM_BONES = 4;
constexpr uint32_t NET_IMAGE_WIDTH = 256;
constexpr uint32_t NET_IMAGE_HEIGHT = 256;
constexpr char MODEL_DEFAULT[] = "../data/models/heatmap-flic-fusion/matlab.prototxt";
constexpr char TRAINED_WEIGHTS_DEFAULT[] = "../data/models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel";

/*
 * Converts the raw data in a Caffe blob into a container of channels.
 *
 * @param [out] channels Container of channels corresponding to data from blob.
 * @param [in] blob Blob containing concatenated multi-channel data.
 * @param [in] width Width of a channel in blob.
 * @param [in] height Height of a channel in blob.
 */
static void
channels_from_blob(std::vector<cv::Mat>& channels,
                   boost::shared_ptr<caffe::Blob<float>> blob,
                   const int32_t width,
                   const int32_t height)
{
        float *raw_data = blob->mutable_cpu_data();
        for (int32_t channel_index = 0;
             channel_index < blob->channels();
             ++channel_index) {
                cv::Mat channel{height, width, CV_32FC1, raw_data};
                channels.push_back(channel);
                raw_data += width*height;
        }
}

/*
 * Converts OpenCV BGR format image to 32-bit floating point RGB format and
 * copies the image to the input blob of heatmap_net.
 *
 * @param [out] heatmap_net Network whose input layer should be filled with the
 * RGB pixel data from image.
 * @param [in/out] image The image to serve as input layer to heatmap_net
 */
static void
copy_image_to_input_blob(caffe::Net<float>& heatmap_net, cv::Mat& image)
{
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32FC3);

        std::vector<cv::Mat> input_channels;
        channels_from_blob(input_channels,
                           heatmap_net.blobs().front(),
                           image.cols,
                           image.rows);
        cv::split(image, input_channels);
}

/*
 * Outputs a set of joint locations from the conv5_fusion layer of the network.
 *
 * @param [out] joints The joint locations to be output.
 * @param [in] channel_size Original joint channel size from output layer of
 * heatmap network.
 * @param [in] heatmap_net The heatmap network.
 */
static void
get_joints_from_network(cv::Point *joints,
                        const cv::Size channel_size,
                        const caffe::Net<float>& heatmap_net)
{
        auto heatmap_blob = heatmap_net.blob_by_name("conv5_fusion");
        std::vector<cv::Mat> joints_channels;
        channels_from_blob(joints_channels,
                           heatmap_blob,
                           heatmap_blob->shape(3),
                           heatmap_blob->shape(2));

        assert(joints_channels.size() == NUM_JOINTS);

        for (uint32_t joints_index = 0;
             joints_index < NUM_JOINTS;
             ++joints_index) {
                cv::Mat& joint = joints_channels.at(joints_index);
                cv::Mat joint_resized;
                cv::resize(joint, joint_resized, channel_size);

                cv::minMaxLoc(joint_resized,
                              NULL,
                              NULL,
                              NULL,
                              joints + joints_index);
        }
}

/*
 * Draws an upper-body skeleton on image based on the joint positions given in
 * joints.
 *
 * @param [out] image Image to draw skeleton on.
 * @param [in] joints Set of joint locations: wrists, elbows, shoulders and
 * head.
 */
static void
draw_skeleton(cv::Mat& image, const cv::Point *joints)
{
        constexpr uint32_t BONE_MAP[NUM_BONES][2] = {
                {2, 4},
                {4, 6},
                {1, 3},
                {3, 5}
        };

        for (uint32_t bone_index = 0;
             bone_index < NUM_BONES;
             ++bone_index) {
                cv::line(image,
                         joints[BONE_MAP[bone_index][0]],
                         joints[BONE_MAP[bone_index][1]],
                         CV_RGB(0, 0xff, 0),
                         3);
        }

        for (uint32_t joints_index = 0;
             joints_index < NUM_JOINTS;
             ++joints_index) {
                cv::circle(image,
                           joints[joints_index],
                           5,
                           CV_RGB(0xff, 0, 0),
                           -1);
        }
}

std::unique_ptr<caffe::Net<float>>
init_pose_estimator_network(const std::string& model,
                            const std::string& trained_weights)
{
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        auto heatmap_net =
                std::unique_ptr<caffe::Net<float>>{new caffe::Net<float>{model, caffe::TEST}};

        heatmap_net->CopyTrainedLayersFrom(trained_weights);

        return heatmap_net;
}

void image_pose_overlay(caffe::Net<float>& heatmap_net, cv::Mat& image)
{
        cv::Size image_original_size = cv::Size{image.cols, image.rows};
        cv::resize(image, image, cv::Size{NET_IMAGE_WIDTH, NET_IMAGE_HEIGHT});

        copy_image_to_input_blob(heatmap_net, image);

        heatmap_net.Forward();

        cv::Point joints[NUM_JOINTS];
        get_joints_from_network(joints,
                                cv::Size{image.cols, image.rows},
                                heatmap_net);

        image.convertTo(image, CV_8UC3);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        draw_skeleton(image, joints);

        cv::resize(image, image, image_original_size);
}

extern "C" int32_t estimate_pose_from_c(void *image,
					uint32_t *size_bytes,
					uint32_t max_size_bytes);

int32_t estimate_pose_from_c(void *image, uint32_t *size_bytes, uint32_t max_size_bytes)
{
        if (image == NULL)
                return ESTIMATE_POSE_ERROR;

	try {
		std::unique_ptr<caffe::Net<float>> heatmap_net =
			init_pose_estimator_network(std::string{MODEL_DEFAULT},
						    std::string{TRAINED_WEIGHTS_DEFAULT});

		cv::InputArray image_input_array{image, static_cast<int32_t>(*size_bytes)};

		cv::Mat image_mat = imdecode(image_input_array, cv::IMREAD_COLOR);

		image_pose_overlay(*heatmap_net, image_mat);

		std::vector<int> compression_params;
		compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(9);

		std::vector<uint8_t> posed_image_png;
		cv::imencode(".png",
			     cv::InputArray{image_mat},
			     posed_image_png,
			     compression_params);

		if (posed_image_png.size() > max_size_bytes) {
			fprintf(stderr,
				"estimate_pose_from_c input buffer size too small\n"
				"require %lu but got %d\n",
				posed_image_png.size(),
				max_size_bytes);

			*size_bytes = posed_image_png.size();

			return ESTIMATE_POSE_ERROR;
		}

		std::memcpy(image, &posed_image_png[0], posed_image_png.size());

		return posed_image_png.size();
	} catch (std::exception e) {
		fprintf(stderr, "estimate_pose_from_c exception: %s\n",
			e.what());
		return ESTIMATE_POSE_ERROR;
	}
}
