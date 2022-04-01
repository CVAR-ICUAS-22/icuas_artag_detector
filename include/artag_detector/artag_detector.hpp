/*!********************************************************************************
 * \brief     Differential Flatness controller Implementation
 * \authors   Miguel Fernandez-Cortizas
 * \copyright Copyright (c) 2020 Universidad Politecnica de Madrid
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

#ifndef __ARTAG_DETECTOR_H__
#define __ARTAG_DETECTOR_H__

//  ros
#include <sys/wait.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
// #include <tf2_ros/transform_broadcaster.h>
// #include <tf2_ros/static_transform_broadcaster.h>
// #include <tf2_ros/buffer.h>
// #include <tf2_ros/transform_listener.h>

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>

#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/TwistStamped.h"
#include "geometry_msgs/TwistWithCovarianceStamped.h"
#include "image_transport/image_transport.h"
#include "mav_msgs/RateThrust.h"
#include "nav_msgs/Odometry.h"
#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "trajectory_msgs/MultiDOFJointTrajectory.h"

// Eigen
#include <Eigen/Dense>

// OpenCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// Std libraries
#include <math.h>

#include <array>
#include <iostream>
#include <vector>

// definitions

#define ODOM_TOPIC "odometry"
#define ODOM_TOPIC_TYPE nav_msgs::Odometry
#define RGB_IMAGE_TOPIC "camera/color/image_raw"
#define RGB_IMAGE_TOPIC_TYPE sensor_msgs::Image
#define RGB_CAMERA_INFO_TOPIC "camera/color/camera_info"
#define RGB_CAMERA_INFO_TOPIC_TYPE sensor_msgs::CameraInfo
#define ARTAG_POSE_TOPIC "artag_pose"
#define ARTAG_POSE_TOPIC_TYPE geometry_msgs::PoseStamped
#define CAMERA_FRAME "camera"
#define REF_FRAME "world"

#define DEBUG 1
#define SATURATE_YAW_ERROR 1
#define SPEED_GAIN 1

class ArTagDetector
{
private:
  ros::NodeHandle nh_;

  ros::Subscriber sub_odom_;
  ros::Subscriber sub_speed_reference_;
  ros::Publisher pub_artag_pose_;
  // create image_transport
  image_transport::ImageTransport it_;
  image_transport::Subscriber sub_rgb_image_;
  // obtain camera_info
  ros::Subscriber sub_camera_info_;
  bool has_camera_info_ = false;

  Eigen::Vector3d relative_artag_position_;
  geometry_msgs::PoseStamped odom_pose_;

  cv::Mat rgb_image_;
  sensor_msgs::CameraInfo camera_info_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener *tf_listener_;

public:
  ArTagDetector();
  ~ArTagDetector() { cv::destroyAllWindows(); };

  void addPropellersMask(cv::Mat &image)
  {
    int top_row = image.rows / 2 - image.rows / 3;
    int bottom_row = image.rows / 2 - image.rows / 10;
    int left_col = image.cols / 2 - image.cols / 3;
    int right_col = image.cols / 2 + image.cols / 3;
    cv::rectangle(image, cv::Point(0, top_row), cv::Point(left_col, bottom_row),
                  cv::Scalar(255, 255, 255), -1);
    cv::rectangle(image, cv::Point(right_col, top_row), cv::Point(image.cols, bottom_row),
                  cv::Scalar(255, 255, 255), -1);
  };

  geometry_msgs::PoseStamped convertRelativePoseToAbsolutePose(
      geometry_msgs::PoseStamped relative_pose, geometry_msgs::PoseStamped odom_pose)
  {
    // relative pose is in camera frame, odom pose is in world frame
    geometry_msgs::PoseStamped absolute_pose;
    absolute_pose.header = relative_pose.header;
    absolute_pose.pose.position.x = odom_pose.pose.position.x + relative_pose.pose.position.z + 0.2;
    absolute_pose.pose.position.y = odom_pose.pose.position.y - relative_pose.pose.position.x;
    absolute_pose.pose.position.z = odom_pose.pose.position.z - relative_pose.pose.position.y + 0.05;
    return absolute_pose;
  }

  void getArTagPositionFromPnP(cv::Rect &ar_tag_rect,
                               sensor_msgs::CameraInfo &camera_info)
  {
    // get the ar tag corners from the rectangle
    std::vector<cv::Point2f> ar_tag_corners;
    ar_tag_corners.push_back(cv::Point2f(ar_tag_rect.x, ar_tag_rect.y));
    ar_tag_corners.push_back(cv::Point2f(ar_tag_rect.x + ar_tag_rect.width, ar_tag_rect.y));
    ar_tag_corners.push_back(
        cv::Point2f(ar_tag_rect.x + ar_tag_rect.width, ar_tag_rect.y + ar_tag_rect.height));
    ar_tag_corners.push_back(cv::Point2f(ar_tag_rect.x, ar_tag_rect.y + ar_tag_rect.height));

    const double square_size = 0.2f;
    std::vector<cv::Point3f> ar_obj_points;
    ar_obj_points.push_back(cv::Point3f(-square_size / 2, square_size / 2, 0));
    ar_obj_points.push_back(cv::Point3f(square_size / 2, square_size / 2, 0));
    ar_obj_points.push_back(cv::Point3f(square_size / 2, -square_size / 2, 0));
    ar_obj_points.push_back(cv::Point3f(-square_size / 2, -square_size / 2, 0));

    // get the camera matrix
    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << camera_info.K.at(0), camera_info.K.at(1), camera_info.K.at(2),
         camera_info.K.at(3), camera_info.K.at(4), camera_info.K.at(5), camera_info.K.at(6),
         camera_info.K.at(7), camera_info.K.at(8));
    // get the distortion coefficients
    cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << camera_info.D.at(0), camera_info.D.at(1),
                           camera_info.D.at(2), camera_info.D.at(3), camera_info.D.at(4));
    // get the rotation and translation vectors
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1); // output rotation vector
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    // solve pnp using P3P
    cv::solvePnP(ar_obj_points, ar_tag_corners, camera_matrix, dist_coeffs, rvec, tvec);

    static geometry_msgs::PoseStamped last_artag_pose_sent;
    static geometry_msgs::PoseStamped last_artag_pose;
    geometry_msgs::PoseStamped artag_relative_pose;
    geometry_msgs::PoseStamped artag_global_pose;
    geometry_msgs::PoseStamped new_artag_pose;

    try
    {
      artag_relative_pose.header.frame_id = REF_FRAME;
      artag_relative_pose.header.stamp = ros::Time::now();
      artag_relative_pose.pose.position.x = tvec.at<double>(0);
      artag_relative_pose.pose.position.y = tvec.at<double>(1);
      artag_relative_pose.pose.position.z = tvec.at<double>(2);
      artag_relative_pose.pose.orientation.x = rvec.at<double>(0);
      artag_relative_pose.pose.orientation.y = rvec.at<double>(1);
      artag_relative_pose.pose.orientation.z = rvec.at<double>(2);
      artag_relative_pose.pose.orientation.w = rvec.at<double>(3);

      auto pose_transform = tf_buffer_.lookupTransform(REF_FRAME, addNamespace(CAMERA_FRAME), ros::Time(0));
      tf2::doTransform(artag_relative_pose, artag_global_pose, pose_transform);

      applyDetectionConstrains(artag_global_pose);

      if (artag_global_pose.pose.position.x > 12.5)
        artag_global_pose.pose.position.x = 12.5;
      if (artag_global_pose.pose.position.y > 7.5)
        artag_global_pose.pose.position.y = 7.5;
      if (artag_global_pose.pose.position.y < -7.5)
        artag_global_pose.pose.position.y = -7.5;
      if (artag_global_pose.pose.position.z < 0.5)
        artag_global_pose.pose.position.z = 0.5;

      float alpha = 0.5;
      // moving average
      new_artag_pose.pose.position.x = alpha * artag_global_pose.pose.position.x +
                                       (1 - alpha) * last_artag_pose.pose.position.x;
      new_artag_pose.pose.position.y = alpha * artag_global_pose.pose.position.y +
                                       (1 - alpha) * last_artag_pose.pose.position.y;
      new_artag_pose.pose.position.z = alpha * artag_global_pose.pose.position.z +
                                       (1 - alpha) * last_artag_pose.pose.position.z;

      float distance = sqrt(pow((new_artag_pose.pose.position.x - last_artag_pose_sent.pose.position.x), 2) +
                            pow((new_artag_pose.pose.position.y - last_artag_pose_sent.pose.position.y), 2) +
                            pow((new_artag_pose.pose.position.z - last_artag_pose_sent.pose.position.z), 2));

      if (distance > 0.1)
      {
        // ROS_WARN("Distance between ARtag estimation: %f", distance);
        ROS_INFO("ARTag position: %f %f %f", new_artag_pose.pose.position.x,
                 new_artag_pose.pose.position.y, new_artag_pose.pose.position.z);
        pub_artag_pose_.publish(new_artag_pose);
        last_artag_pose_sent = new_artag_pose;
      }

      last_artag_pose = new_artag_pose;
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("Transform fail. Failure %s", ex.what());
      ros::Duration(1.0).sleep();
    }
  };

  cv::Mat processImage(const cv::Mat &image)
  {
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    addPropellersMask(image_gray);
    // cv::imshow("image_gray", image_gray);

    cv::threshold(image_gray, image_gray, 100, 255, cv::THRESH_BINARY_INV);
    cv::dilate(image_gray, image_gray, cv::Mat(), cv::Point(-1, -1), 2);
    cv::dilate(image_gray, image_gray, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(image_gray, image_gray, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(image_gray, image_gray, cv::Mat(), cv::Point(-1, -1), 2);

    // findContours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image_gray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0));

    // draw biggest contour as a rectangle
    cv::Rect bounding_rect;
    if (contours.size() > 0)
    {
      std::vector<cv::Point> biggest_contour = contours[0];
      for (int i = 1; i < contours.size(); i++)
      {
        if (contours[i].size() > biggest_contour.size())
        {
          biggest_contour = contours[i];
        }
      }
      bounding_rect = cv::boundingRect(biggest_contour);
      if (has_camera_info_)
      {
        getArTagPositionFromPnP(bounding_rect, camera_info_);
      }
    }
    cv::Mat image_rect = image.clone();
    cv::cvtColor(image_rect, image_rect, cv::COLOR_BGR2RGB);
    cv::rectangle(image_rect, bounding_rect, cv::Scalar(0, 255, 0), 2);
    cv::imshow("image_rect", image_rect);
    cv::waitKey(1);
    return image_rect;
  }

  void run()
  {
    // if there is a rgb image show it
    if (!rgb_image_.empty())
    {
      processImage(rgb_image_);
    }
  };

private:
  void CallbackOdomTopic(const nav_msgs::Odometry &odom_msg);
  // void CallbackSpeedRefTopic(const geometry_msgs::TwistStamped& twist_msg);
  void CallbackRgbImageTopic(const sensor_msgs::ImageConstPtr &rgb_image_msg);
  void CallbackCameraInfoTopic(const sensor_msgs::CameraInfoConstPtr &camera_info_msg);

  void applyDetectionConstrains(geometry_msgs::PoseStamped &pose)
  {
    if (pose.pose.position.x > 12.5)
      pose.pose.position.x = 12.5;
    if (pose.pose.position.y > 7.5)
      pose.pose.position.y = 7.5;
    if (pose.pose.position.y < -7.5)
      pose.pose.position.y = -7.5;
    if (pose.pose.position.z < 0.5)
      pose.pose.position.z = 0.5;
    // if (abs(pose.pose.position.y) < 7.5)
    //   pose.pose.position.z = 1.5;
  }

  std::string addNamespace(const std::string name)
  {
    std::string ns = nh_.getNamespace();
    if (ns.empty())
    {
      return name;
    }
    else
    {
      if (ns[0] == '/')
      {
        ns = ns.substr(1);
      }
      return ns + "/" + name;
    }
  }
};

#endif
