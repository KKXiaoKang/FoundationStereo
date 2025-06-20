import rospy
from sensor_msgs.msg import CameraInfo

def camera_info_callback(msg):
    fx = msg.P[0]        # 右目相机的fx
    tx = msg.P[3]        # -fx * baseline
    baseline = abs(tx / fx)
    print(f"Baseline: {baseline:.4f} meters")

rospy.init_node('get_baseline')
sub = rospy.Subscriber('/zedm/zed_node/right/camera_info', 
                      CameraInfo, 
                      camera_info_callback)
rospy.spin()