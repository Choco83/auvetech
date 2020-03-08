# Traffic Light detecion
Test task for Auvetech

## How to run
This package assumes you have already installed working version of PyTorch.
This package is taken from [YOLOv3-ROS](https://github.com/yehengchen/YOLOv3-ROS/tree/master/yolov3_pytorch_ros) repo and modified according to task.
You need to clone this package in your src directory and build it.

You need to make python nodes executable, for that run,
```
$ sh make_exec.sh
```
## Downloading models
In model folder download_weights.sh can be run to download pretrained model weights

## Running node
```
$ roslaunch yolov3_pytorch_ros detector.launch
```
This will launch both the required nodes, fetcher and analysis with a saved video in video folder.