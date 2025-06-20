#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import Point
import message_filters

import argparse
import os
import sys

code_dir = os.path.dirname(os.path.realpath(__file__))
try:
    # We are in FoundationStereo/scripts/depth_eval, so we need to go up two levels
    sys.path.append(os.path.join(code_dir, '..', '..'))
    from Utils import vis_disparity
except ImportError as e:
    rospy.logwarn(f"Could not import vis_disparity, will use a fallback. Error: {e}")
    vis_disparity = None


class DepthEvaluator:
    def __init__(self, args):
        self.bridge = CvBridge()
        self.K = None
        self.baseline = None
        self.original_height = 0
        self.original_width = 0
        self.args = args

        if self.args.use_camera_info:
            if not self.load_intrinsics_from_topic():
                rospy.logerr("Shutting down due to failure to get intrinsics from topic.")
                rospy.signal_shutdown("Failed to load intrinsics from topic")
                return
        else:
            if not self.load_intrinsics_from_file():
                rospy.logerr("Shutting down due to failure to get intrinsics from file.")
                rospy.signal_shutdown("Failed to load intrinsics from file")
                return

        self.disparity_sub = message_filters.Subscriber('/foundation_stereo/disparity/depth', Image)
        self.tag_sub = message_filters.Subscriber('/tag_detections', AprilTagDetectionArray)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.disparity_sub, self.tag_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.callback)

        self.vis_pub = rospy.Publisher('/depth_eval/roi_visualization', Image, queue_size=1)
        self.data_pub = rospy.Publisher('/depth_eval/eval_data', Point, queue_size=10)
        
        rospy.loginfo("Depth evaluator node started and waiting for messages.")

    def load_intrinsics_from_file(self):
        try:
            with open(self.args.intrinsic_file, 'r') as f:
                lines = f.readlines()
                self.K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
                self.baseline = float(lines[1])
                rospy.loginfo(f"Intrinsics loaded from {self.args.intrinsic_file}")
                return True
        except Exception as e:
            rospy.logerr(f"Failed to load intrinsics file at {self.args.intrinsic_file}: {e}")
            return False

    def load_intrinsics_from_topic(self):
        try:
            rospy.loginfo("Waiting for camera info from topics...")
            left_cam_info = rospy.wait_for_message(self.args.left_cam_info_topic, CameraInfo, timeout=10.0)
            self.K = np.array(left_cam_info.K).reshape(3,3)
            self.original_height = left_cam_info.height
            self.original_width = left_cam_info.width
            rospy.loginfo(f"Got K and original dimensions ({self.original_width}x{self.original_height}) from {self.args.left_cam_info_topic}")

            right_cam_info = rospy.wait_for_message(self.args.right_cam_info_topic, CameraInfo, timeout=10.0)
            Tx = right_cam_info.P[3]
            fx = right_cam_info.P[0]
            if fx == 0:
                rospy.logerr("fx is zero in right camera projection matrix, cannot calculate baseline.")
                return False
            self.baseline = -Tx / fx
            rospy.loginfo(f"Got baseline ({self.baseline:.4f}m) from {self.args.right_cam_info_topic}")
            
            return True
        except (rospy.exceptions.ROSException, rospy.exceptions.ROSInterruptException) as e:
            rospy.logerr(f"Failed to get camera info from topic: {e}")
            return False

    def callback(self, disparity_msg, tag_msg):
        if not tag_msg.detections:
            return

        try:
            disparity_map = self.bridge.imgmsg_to_cv2(disparity_msg, desired_encoding='32FC1')
        except Exception as e:
            rospy.logerr(f"Error processing disparity image: {e}")
            return
        
        # --- BEGIN: Automatic Scaling Correction ---
        disparity_height, disparity_width = disparity_map.shape[:2]
        scale = 1.0

        if self.original_height > 0:
            scale = disparity_height / self.original_height
        elif self.args.use_camera_info:
            rospy.logwarn_throttle(5, "Original camera dimensions not available, cannot determine scale. Assuming scale=1.0.")
        
        # 打印 scale
        rospy.loginfo(f"scale: {scale}")

        # Create a scaled version of the intrinsics for this specific message
        K_scaled = self.K.copy()
        if scale != 1.0:
            K_scaled[:2, :] *= scale
        # --- END: Automatic Scaling Correction ---
        
        detection = tag_msg.detections[0]
        
        pos = detection.pose.pose.pose.position
        tx, ty, tz = pos.x, pos.y, pos.z
        
        tag_size = detection.size[0]
        
        if tz <= 0:
            rospy.logwarn_throttle(5, "Tag is behind or on the camera plane.")
            return

        fx, fy, cx, cy = K_scaled[0,0], K_scaled[1,1], K_scaled[0,2], K_scaled[1,2]

        u_c = int(fx * tx / tz + cx)
        v_c = int(fy * ty / tz + cy)

        width_px = int(fx * tag_size / tz)
        height_px = int(fy * tag_size / tz)

        x1 = max(0, u_c - width_px // 2)
        y1 = max(0, v_c - height_px // 2)
        x2 = min(disparity_map.shape[1], u_c + width_px // 2)
        y2 = min(disparity_map.shape[0], v_c + height_px // 2)

        if x1 >= x2 or y1 >= y2:
            rospy.logwarn_throttle(5, "ROI is out of bounds or invalid.")
            return

        disparity_roi = disparity_map[y1:y2, x1:x2]
        
        if disparity_roi.size == 0:
            rospy.logwarn_throttle(5, "ROI has zero size.")
            return

        with np.errstate(divide='ignore'):
            depth_roi = self.baseline * fx / disparity_roi
        
        depth_roi = depth_roi[np.isfinite(depth_roi)]
        depth_roi = depth_roi[(depth_roi > 0.1) & (depth_roi < 10.0)]

        mean_depth = 0.0 # 平均深度
        if depth_roi.size > 0:
            mean_depth = np.mean(depth_roi)
        else:
            rospy.logwarn_throttle(5, "No valid depth points in ROI.")

        rospy.loginfo(f"Tag at ({tx:.2f}, {ty:.2f}, {tz:.2f})m, estimated avg depth in ROI: {mean_depth:.2f}m")

        # --- BEGIN: Publish evaluation data ---
        eval_data_msg = Point()
        eval_data_msg.x = tz # Ground truth distance from tag
        eval_data_msg.y = mean_depth # Estimated depth from disparity
        eval_data_msg.z = tz - mean_depth # The error
        self.data_pub.publish(eval_data_msg)
        # --- END: Publish evaluation data ---

        if vis_disparity:
            vis_img = vis_disparity(disparity_map)
            if len(vis_img.shape) == 2 or vis_img.shape[2] == 1:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        else:
            disp_norm = cv2.normalize(disparity_map, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
            vis_img = cv2.cvtColor(disp_norm, cv2.COLOR_GRAY2RGB)
        
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"Avg Depth: {mean_depth:.2f}m"
        cv2.putText(vis_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "rgb8")
            vis_msg.header = disparity_msg.header
            self.vis_pub.publish(vis_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing visualization: {e}")


if __name__ == "__main__":
    rospy.init_node("depth_eval")

    parser = argparse.ArgumentParser(description="Evaluate depth from disparity and AprilTags.")
    
    # File-based intrinsics arguments
    default_intrinsic_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'zed', 'data', 'K.txt')
    parser.add_argument('--intrinsic_file', type=str, 
                        default=default_intrinsic_path,
                        help='Path to camera intrinsic matrix and baseline file.')
    
    # Topic-based intrinsics arguments
    parser.add_argument('--use_camera_info', action='store_true', 
                        help='Use camera_info topic for intrinsics instead of file.')
    parser.add_argument('--left_cam_info_topic', type=str, default='/zedm/zed_node/left/camera_info',
                        help='Left camera info topic.')
    parser.add_argument('--right_cam_info_topic', type=str, default='/zedm/zed_node/right/camera_info',
                        help='Right camera info topic.')

    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    evaluator = DepthEvaluator(args)
    if not rospy.is_shutdown():
        rospy.spin()