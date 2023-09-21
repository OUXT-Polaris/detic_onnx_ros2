# detic_onnx_ros2 

## What is it?

Running detic instance segmentation with ROS 2 topic.

[![](https://img.youtube.com/vi/EJEPW2xSSVs/0.jpg)](https://www.youtube.com/watch?v=EJEPW2xSSVs)

## How to use.

In order to build this package, just run.

```
rosdep install -iry --from-paths .
cd (workspace root)
colcon build --symlink-install
```

## How to run.

```
ros2 run detic_onnx_ros2 detic_onnx_ros2_node --ros-args --remap image_raw:=(input image topic)
```

You can see detection results in `/detic_result/image` topic with sensor_msgs/msg/Image type.

## Roadmap
- [x] Inference with ROS 2 sensor_msgs/msg/Image topic.
- [x] Visualize segmentation result.
- [x] Publish object class.
- [x] Publish object score.
- [ ] Add launch file.
- [ ] Add config file for setting detection width / detic model type / vocaburary etc...
- [ ] Publish object mask.
- [ ] Inference with GPU.
- [ ] Add test case.

## Limitation
Custom vocabulary will not be supported because of onnx model used in this package does not support it.
