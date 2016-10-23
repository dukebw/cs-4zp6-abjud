#ifndef _ESTIMATE_POSE_H_
#define _ESTIMATE_POSE_H_

#include "caffe/net.hpp"
#include <memory>

std::unique_ptr<caffe::Net<float>>
init_pose_estimator_network(const std::string& model,
                            const std::string& trained_weights);

void image_pose_overlay(caffe::Net<float>& heatmap_net, cv::Mat& image);

#endif // _ESTIMATE_POSE_H_
