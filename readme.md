# ZED相机官网下载序列号cali校正文件
### 1）下载cali校正文件
* 其中的SN通过`roslaunch zed_wrapper zedm.launch`可以读取到
```bash
http://calib.stereolabs.com/?SN=13128538

# 下载文件名
SNXXXXX.conf
```
### 2） 将文件放入
```bash
On Linux : /usr/local/zed/settings/
On Windows : C:\ProgramData\Stereolabs\settings

# 对于linux来说
cp ~/SLAM/config/zed/SN13128538.conf /usr/local/zed/settings
```


### 启动
#### 1）启动双目zedm双目相机
```bash
roslaunch zed_wrapper zedm.launch
```

#### 2） 启动foundationstereo处理
```bash
# 图片演示
python3 scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/

# full version - 2.0Hz 4090D i9-13900KF
python3 scripts/run_demo_topic.py --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/ --vis_mode publish

# fast version - 2.5Hz 4090D i9-13900KF
python3 scripts/run_demo_topic.py --ckpt_dir ./pretrained_models/11-33-40/model_best_bp2.pth --out_dir ./test_outputs/ --vis_mode publish
```

### 验证tag码的深度信息 和 FoudationStereo的深度 之间的损失
```bash
# 启动tag码信息，请注意检查tag码的tag.yaml的尺寸信息
roslaunch apriltag_zedm_pkg apriltag_zedm_enable.launch

# 启动深度验证 | 实时查看信息
cd your_workspace_path/FoundationStereo
python3 scripts/depth_eval/depth_eval.py --use_camera_info
```
* ![depth_eval](./IMG/depth_eval.jpg)