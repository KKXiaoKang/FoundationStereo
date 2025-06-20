# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import logging
import argparse

DEBUG_MODE = False
FIRST_FRAME_FLAG = True

class ROS_ZED_CAMERA:
    def __init__(self):
      left_image_sub = rospy.Subscriber('/zedm/zed_node/left/image_rect_color', Image, self.left_image_callback)
      right_image_sub = rospy.Subscriber('/zedm/zed_node/right/image_rect_color', Image, self.right_image_callback)
      self.disp_pub = rospy.Publisher('/foundation_stereo/disparity/depth', Image, queue_size=1)
      self.vis_pub = rospy.Publisher('/foundation_stereo/visualization/vis', Image, queue_size=1)
      self.left_image = None
      self.right_image = None
      self.left_msg_header = None
      self.bridge = CvBridge()

    def left_image_callback(self, msg):
      if DEBUG_MODE:
        rospy.loginfo(" left image callback")
      self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
      self.left_msg_header = msg.header

    def right_image_callback(self, msg):
      if DEBUG_MODE:
        rospy.loginfo(" right image callback")
      self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

if __name__=="__main__":
  # 初始化节点
  rospy.init_node('zed_camera_foundation_stereo_node')

  # 实例化对象
  ros_zed_camera = ROS_ZED_CAMERA()
  time.sleep(2)

  # --- 参数解析 (精简后) ---
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser(description="Real-time stereo depth estimation from ROS topics.")
  
  # -- 模型与配置 --
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, 
                      help='Path to the pretrained model checkpoint file.')
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../zed/data/K.txt', type=str, 
                      help='Path to the camera intrinsics file (K matrix and baseline). Used only if --get_pc is enabled.')

  # -- 性能与质量控制 --
  # 图像缩放比例。这是一个非常重要的性能调节旋钮。值越小（如0.5），处理速度越快，但会牺牲细节和精度
  parser.add_argument('--scale', default=1, type=float, 
                      help='Downscaling factor for input images. Must be <=1. Smaller values (e.g., 0.5) increase speed but reduce accuracy.')
  # 模型内部的迭代优化次数。次数越少，推理速度越快，但可能导致深度图质量下降，边缘不清晰。
  parser.add_argument('--valid_iters', type=int, default=32, 
                      help='Number of iterative updates in the model\'s forward pass. Fewer iterations speed up inference but may decrease quality.')
  # 是否启用分层推理模式。对于超高分辨率图像（大于1K），此模式可以有效管理显存并提高性能。对于普通分辨率，通常保持关闭。
  parser.add_argument('--hiera', default=0, type=int, 
                      help='Enable hierarchical inference. Needed for high-resolution (>1K) images to manage memory and performance.')

  # -- 后处理与可视化 --
  # 控制结果的输出方式。publish 表示发布到ROS话题，save 表示存为图片，both 表示两者都做。对于实时应用，通常设为 publish。
  parser.add_argument('--vis_mode', default='publish', choices=['publish', 'both', 'save'], 
                      help='Visualization mode: "publish" to a ROS topic, "save" to a file, or "both".')
  # 是否生成三维点云。警告：这是一个非常消耗计算资源的操作，会显著降低处理帧率。仅在需要3D数据时开启。
  parser.add_argument('--get_pc', type=int, default=0, 
                      help='Enable point cloud generation (set to 1). Warning: This is computationally expensive.')
  # （点云参数）生成点云时，保留的最大深度值（单位：米）。
  parser.add_argument('--z_far', default=10, type=float, 
                      help='Maximum depth (in meters) to include in the point cloud.')
  # （点云参数）移除在双目视图中没有重叠观测的点。这能有效过滤掉不可靠的边缘点，提升点云质量。
  parser.add_argument('--remove_invisible', default=1, type=int, 
                      help='Remove points from the cloud that are not visible in both stereo views, increasing reliability.')

  # （点云参数）是否对生成的点云进行去噪处理。
  parser.add_argument('--denoise_cloud', type=int, default=1, 
                      help='Enable radius outlier removal to denoise the point cloud.')
  
  # （点云去噪参数）半径滤波时，用于判断一个点是否为离群点的邻域点数量。
  parser.add_argument('--denoise_nb_points', type=int, default=30, 
                      help='Number of neighboring points to consider for outlier removal.')
  # （点云去噪参数）半径滤波时，用于判断一个点是否为离群点的邻域半径（单位：米）。
  parser.add_argument('--denoise_radius', type=float, default=0.03, 
                      help='The radius used for outlier removal.')
  
  args, _ = parser.parse_known_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  # os.makedirs(args.out_dir, exist_ok=True)
 
  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  # 初始化模型
  model = FoundationStereo(args)

  # 加载checkpoint权重
  ckpt = torch.load(ckpt_dir)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  # 模型加载到GPU
  model.cuda()
  model.eval()

  # 代码路径
  code_dir = os.path.dirname(os.path.realpath(__file__))

  logging.info("Starting processing loop...")
  while not rospy.is_shutdown():
    # ---- Cycle Start ----
    start_cycle_time = time.time()

    # 等待双目图像
    if ros_zed_camera.left_image is None or ros_zed_camera.right_image is None or ros_zed_camera.left_msg_header is None:
      rospy.loginfo_throttle(1.0, "Waiting for stereo images...")
      rospy.sleep(0.1)
      continue

    # 1. 获取图像 & CPU预处理
    start_preprocess_time = time.time()
    img0 = ros_zed_camera.left_image
    img1 = ros_zed_camera.right_image
    scale = args.scale
    assert scale<=1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    preprocess_time = time.time() - start_preprocess_time
    
    # 打印图像尺寸
    if DEBUG_MODE or FIRST_FRAME_FLAG:
      logging.info(f"img0: {img0.shape}")
      logging.info(f"img1: {img1.shape}")
      FIRST_FRAME_FLAG = False
    
    # 2. 图像加载到GPU & Pad处理
    start_gpu_pad_time = time.time()
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    gpu_pad_time = time.time() - start_gpu_pad_time

    # 3. 深度图推理 (核心步骤)
    start_inference_time = time.time()
    with torch.cuda.amp.autocast(True):
      if not args.hiera:
        disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
      else:
        disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    torch.cuda.synchronize() # 等待GPU操作完成以获得准确计时
    inference_time = time.time() - start_inference_time

    # 4. 深度图解码 (GPU to CPU)
    start_decode_time = time.time()
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    decode_time = time.time() - start_decode_time

    # 5. 可视化图像生成
    start_viz_time = time.time()
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    viz_time = time.time() - start_viz_time
    
    # 6. 消息发布 & 存储
    start_publish_time = time.time()
    save_time = 0.0
    if args.vis_mode in ['save', 'both']:
        start_save_time = time.time()
        imageio.imwrite(f'{args.out_dir}/vis.png', vis)
        save_time = time.time() - start_save_time

    if args.vis_mode in ['publish', 'both']:
        vis_msg = ros_zed_camera.bridge.cv2_to_imgmsg(vis, encoding="rgb8")
        vis_msg.header = ros_zed_camera.left_msg_header
        ros_zed_camera.vis_pub.publish(vis_msg)
    
    disp_msg = ros_zed_camera.bridge.cv2_to_imgmsg(disp.astype(np.float32), encoding="32FC1")
    disp_msg.header = ros_zed_camera.left_msg_header
    ros_zed_camera.disp_pub.publish(disp_msg)
    publish_time = time.time() - start_publish_time - save_time # 从发布时间中排除文件保存时间
    
    if DEBUG_MODE:
      logging.info(f"Output saved to {args.out_dir}")

    # 7. (可选) 点云处理
    start_pc_time = time.time()
    if args.remove_invisible:
      yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
      us_right = xx-disp
      invalid = us_right<0
      disp[invalid] = np.inf

    if args.get_pc:
      with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
      K[:2] *= scale
      depth = K[0,0]*baseline/disp
      np.save(f'{args.out_dir}/depth_meter.npy', depth)
      xyz_map = depth2xyzmap(depth, K)
      pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
      keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
      keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
      pcd = pcd.select_by_index(keep_ids)

      # 保存点云
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
      logging.info(f"PCL saved to {args.out_dir}")

      # 点云去噪
      if args.denoise_cloud:
        logging.info("[Optional step] denoise point cloud...")
        cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
        pcd = inlier_cloud

      # 可视化点云
      logging.info("Visualizing point cloud. Press ESC to exit.")
      vis = o3d.visualization.Visualizer()
      vis.create_window()
      vis.add_geometry(pcd)
      vis.get_render_option().point_size = 1.0
      vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
      vis.run()
      vis.destroy_window()
    pc_time = time.time() - start_pc_time

    # ---- Cycle End & Log Timings ----
    total_cycle_time = time.time() - start_cycle_time
    
    # 格式化耗时日志 (单位: 毫秒 ms)
    timings_log = (
        f"\n--- Frame Timings (ms) ---\n"
        f"  1. CPU Pre-processing : {preprocess_time * 1000:.2f}\n"
        f"  2. GPU Transfer & Pad   : {gpu_pad_time * 1000:.2f}\n"
        f"  3. Model Inference      : {inference_time * 1000:.2f}  (<<-- 核心步骤)\n"
        f"  4. Disparity Decode     : {decode_time * 1000:.2f}\n"
        f"  5. Visualization        : {viz_time * 1000:.2f}\n"
        f"  6. ROS Publish          : {publish_time * 1000:.2f}\n"
        f"     (File Save)        : {save_time * 1000:.2f}\n"
        f"  7. Point Cloud Proc     : {pc_time * 1000:.2f}\n"
        f"----------------------------\n"
        f"  Total Cycle Time      : {total_cycle_time * 1000:.2f} ms\n"
        f"  Estimated FPS         : {1.0 / total_cycle_time:.2f}\n"
        f"============================"
    )
    logging.info(timings_log)

