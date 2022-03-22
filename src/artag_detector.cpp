/*!********************************************************************************
 * \brief     Differential Flatness controller Implementation

 * \copyright Copyright (c) 2022 Universidad Politecnica de Madrid
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/

#include "artag_detector/artag_detector.hpp"

#include "trajectory_msgs/MultiDOFJointTrajectoryPoint.h"

ArTagDetector::ArTagDetector(/* args */) : it_(nh_) {
  // traj_pub_ = nh_.advertise<TRAJECTORY_TOPIC_TYPE>(TRAJECTORY_TOPIC, 1);
  sub_odom_ = nh_.subscribe(ODOM_TOPIC, 1, &ArTagDetector::CallbackOdomTopic, this);
  // sub_speed_reference_ =
  //     nh_.subscribe(SPEED_REFERENCE_TOPIC, 1, &ArTagDetector::CallbackSpeedRefTopic, this);

  // sub rgb image
  sub_rgb_image_ = it_.subscribe(RGB_IMAGE_TOPIC, 1, &ArTagDetector::CallbackRgbImageTopic, this);
  // sub camera_info
  sub_camera_info_ = nh_.subscribe(RGB_CAMERA_INFO_TOPIC, 1, &ArTagDetector::CallbackCameraInfoTopic, this);

  // Artag pose publisher
  pub_artag_pose_ = nh_.advertise<ARTAG_POSE_TOPIC_TYPE>(ARTAG_POSE_TOPIC, 1);
}

/* --------------------------- CALLBACKS ---------------------------*/

void ArTagDetector::CallbackOdomTopic(const ODOM_TOPIC_TYPE& odom_msg) {
  odom_pose_.header = odom_msg.header;
  odom_pose_.pose = odom_msg.pose.pose;
}

// void ArTagDetector::CallbackSpeedRefTopic(const SPEED_REFERENCE_TOPIC_TYPE& speed_reference_msg) {
//   const geometry_msgs::Twist& speed_msg = speed_reference_msg.twist;
// }

void ArTagDetector::CallbackRgbImageTopic(const sensor_msgs::ImageConstPtr& rgb_image_msg) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(rgb_image_msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  // pass image to cv::Mat
  rgb_image_ = cv_ptr->image;
  // cv::imshow("rgb_image", cv_ptr->image);
  // cv::waitKey(3);
}

void ArTagDetector::CallbackCameraInfoTopic(const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
  camera_info_ = *camera_info_msg;
  has_camera_info_ = true;
}

