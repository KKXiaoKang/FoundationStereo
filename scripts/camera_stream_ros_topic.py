#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class StereoCameraNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('stereo_camera_node', anonymous=True)
        
        # 创建发布者
        self.left_pub = rospy.Publisher('/camera/left/image_raw', Image, queue_size=10)
        self.right_pub = rospy.Publisher('/camera/right/image_raw', Image, queue_size=10)
        
        # 创建CV bridge
        self.bridge = CvBridge()
        
        # 从参数服务器获取摄像头参数
        self.camera_id = rospy.get_param('~camera_id', 0)  # 默认使用摄像头0
        self.frame_width = rospy.get_param('~frame_width', 1280)
        self.frame_height = rospy.get_param('~frame_height', 720)
        self.fps = rospy.get_param('~fps', 30)
        
        # 尝试不同的摄像头后端
        backends = [
            cv2.CAP_V4L2,  # 首先尝试V4L2
            cv2.CAP_ANY    # 如果失败则尝试其他后端
        ]
        
        self.cap = None
        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_id, backend)
            if self.cap.isOpened():
                rospy.loginfo(f"成功使用后端 {backend} 打开摄像头")
                break
                
        if not self.cap or not self.cap.isOpened():
            rospy.logerr("无法打开摄像头！请检查：")
            rospy.logerr("1. 摄像头权限是否正确 (sudo usermod -a -G video $USER)")
            rospy.logerr("2. 摄像头是否被其他程序占用")
            rospy.logerr("3. 摄像头ID是否正确")
            return
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 验证实际设置的参数
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        rospy.loginfo(f"摄像头参数设置：")
        rospy.loginfo(f"分辨率: {actual_width}x{actual_height}")
        rospy.loginfo(f"帧率: {actual_fps}")
        
    def run(self):
        rate = rospy.Rate(self.fps)
        
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("无法读取摄像头画面！")
                break
                
            # 假设双目图像是并排的，需要分割成左右两个图像
            height, width = frame.shape[:2]
            mid = width // 2
            
            # 分割左右图像
            left_img = frame[:, :mid]
            right_img = frame[:, mid:]
            
            try:
                # 转换为ROS消息并发布
                left_msg = self.bridge.cv2_to_imgmsg(left_img, "bgr8")
                right_msg = self.bridge.cv2_to_imgmsg(right_img, "bgr8")
                
                self.left_pub.publish(left_msg)
                self.right_pub.publish(right_msg)
                
            except Exception as e:
                rospy.logerr(f"发布图像时出错: {str(e)}")
                
            rate.sleep()
            
    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = StereoCameraNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.cleanup()  