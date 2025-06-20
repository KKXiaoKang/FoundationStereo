#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal, QObject
from geometry_msgs.msg import Point
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# A helper class to emit signals from the ROS callback thread to the main GUI thread
class RosDataEmitter(QObject):
    data_received = pyqtSignal(float, float, float)

    def ros_callback(self, msg):
        self.data_received.emit(msg.x, msg.y, msg.z)

class DepthEvalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Estimation Error Analysis")
        self.setGeometry(100, 100, 1200, 600)

        # --- Data Storage ---
        self.is_sampling = False
        self.timestamps = []
        self.tag_depths = []
        self.estimated_depths = []
        self.errors = []
        
        # --- Main Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Controls Layout (Left Side) ---
        controls_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start Sampling")
        self.btn_start.clicked.connect(self.start_sampling)
        self.btn_stop = QPushButton("Stop Sampling")
        self.btn_stop.clicked.connect(self.stop_sampling)
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addStretch()

        # --- Matplotlib Canvas (Right Side) ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(211) # Top plot for depths
        self.ax2 = self.figure.add_subplot(212) # Bottom plot for error

        # --- Add layouts to main layout ---
        self.main_layout.addLayout(controls_layout)
        self.main_layout.addWidget(self.canvas)

        # --- ROS Integration ---
        self.emitter = RosDataEmitter()
        self.emitter.data_received.connect(self.update_data)
        rospy.init_node('depth_eval_gui', anonymous=True)
        rospy.Subscriber('/depth_eval/eval_data', Point, self.emitter.ros_callback)
        
        # Run rospy.spin() in a separate thread
        self.ros_thread = threading.Thread(target=rospy.spin)
        self.ros_thread.daemon = True
        self.ros_thread.start()

    def start_sampling(self):
        # Clear old data
        self.timestamps = []
        self.tag_depths = []
        self.estimated_depths = []
        self.errors = []
        
        self.is_sampling = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        rospy.loginfo("Started sampling depth data.")

    def stop_sampling(self):
        self.is_sampling = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        rospy.loginfo("Stopped sampling. Final statistics:")
        if self.errors:
            errors_np = np.array(self.errors)
            rospy.loginfo(f"  - Samples collected: {len(self.errors)}")
            rospy.loginfo(f"  - Mean Error: {np.mean(errors_np):.4f} m")
            rospy.loginfo(f"  - Std Dev of Error: {np.std(errors_np):.4f} m")
            rospy.loginfo(f"  - RMSE: {np.sqrt(np.mean(errors_np**2)):.4f} m")


    def update_data(self, tag_depth, estimated_depth, error):
        if not self.is_sampling:
            return
            
        self.timestamps.append(rospy.get_time())
        self.tag_depths.append(tag_depth)
        self.estimated_depths.append(estimated_depth)
        self.errors.append(error)

        self.plot_data()

    def plot_data(self):
        # Top Plot: Depths
        self.ax1.clear()
        self.ax1.plot(self.tag_depths, 'r-', label='Tag Distance (Ground Truth)')
        self.ax1.plot(self.estimated_depths, 'b--', label='Estimated Depth')
        self.ax1.set_ylabel('Depth (m)')
        self.ax1.set_title('Depth Comparison')
        self.ax1.legend(loc='best')
        self.ax1.grid(True)
        
        # Bottom Plot: Error
        self.ax2.clear()
        self.ax2.plot(self.errors, 'g-', label='Error (Truth - Estimate)')
        self.ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        self.ax2.set_xlabel('Sample Number')
        self.ax2.set_ylabel('Error (m)')
        self.ax2.set_title('Estimation Error')
        self.ax2.legend(loc='best')
        self.ax2.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        # Cleanly shutdown ROS when the GUI is closed
        rospy.signal_shutdown("GUI closed")
        self.ros_thread.join(1) # Wait a bit for the thread to exit
        event.accept()

if __name__ == '__main__':
    # Make sure you have the required packages
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print("Error: PyQt5 is not installed. Please run: python3 -m pip install pyqt5")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    main_win = DepthEvalGUI()
    main_win.show()
    sys.exit(app.exec_()) 