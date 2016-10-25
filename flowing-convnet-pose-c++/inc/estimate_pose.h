#ifndef _ESTIMATE_POSE_H_
#define _ESTIMATE_POSE_H_

#include "caffe/net.hpp"
#include <memory>

/*
 * Initializes the heatmap regressor network from the Flowing ConvNets paper.
 *
 * @param [in] model The Caffe text file defining the network model.
 * @param [in] trained_weights The binary Caffe-compatible file produced by
 * training model.
 *
 * @return A pointer to the Caffe network, which has been allocated and
 * initialized by this function call.
 */
std::unique_ptr<caffe::Net<float>>
init_pose_estimator_network(const std::string& model,
                            const std::string& trained_weights);

/*
 * Overlays an upper-body human pose estimate on image.
 *
 * @param [in] heatmap_net Heatmap regressor network.
 * @param [in/out] image An image to be pose-overlaid.
 */
void image_pose_overlay(caffe::Net<float>& heatmap_net, cv::Mat& image);

#endif // _ESTIMATE_POSE_H_
