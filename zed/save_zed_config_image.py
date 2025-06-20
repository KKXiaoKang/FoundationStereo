#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import os

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.save_dir = "data"
        self.k_saved = False
        self.left_image = None  # 新增左图缓存
        self.right_image = None  # 新增右图缓存
        
        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # 订阅双目图像和相机信息
        self.left_sub = rospy.Subscriber("/zedm/zed_node/left/image_rect_color", Image, self.left_callback)
        self.right_sub = rospy.Subscriber("/zedm/zed_node/right/image_rect_color", Image, self.right_callback)
        self.cam_info_sub = rospy.Subscriber("/zedm/zed_node/left/camera_info", CameraInfo, self.info_callback)

    def info_callback(self, msg):
        """保存相机内参到K.txt"""
        if not self.k_saved:
            K = msg.K
            with open(os.path.join(self.save_dir, "K.txt"), "w") as f:
                f.write(" ".join(map(str, K[:3])) + "\n")
                f.write(" ".join(map(str, K[3:6])) + "\n")
                f.write(" ".join(map(str, K[6:])))
            self.k_saved = True
            rospy.loginfo("Camera matrix saved to K.txt")

    def save_current_images(self):
        """保存当前缓存的图像"""
        timestamp = rospy.Time.now().to_nsec()
        if self.left_image is not None:
            cv2.imwrite(os.path.join(self.save_dir, f"left_{timestamp}.png"), self.left_image)
            rospy.loginfo(f"Saved left_{timestamp}.png")
        if self.right_image is not None:
            cv2.imwrite(os.path.join(self.save_dir, f"right_{timestamp}.png"), self.right_image)
            rospy.loginfo(f"Saved right_{timestamp}.png")

    def left_callback(self, msg):
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # 更新左图缓存
        except CvBridgeError as e:
            rospy.logerr(e)

    def right_callback(self, msg):
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # 更新右图缓存
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == "__main__":
    rospy.init_node("zed_image_saver")
    saver = ImageSaver()
    
    print("Press Enter to save images (q to quit)...")
    while not rospy.is_shutdown():
        key = input()  # 等待键盘输入
        if key.lower() == 'q':
            break
        saver.save_current_images()  # 触发保存 