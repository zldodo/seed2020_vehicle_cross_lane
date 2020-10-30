# SEED2020 - Vehicle Cross Lane Event Detection
This repo is developed for the competition of SEED2020 Vehicle Cross Lane Event Detection.

# Install
``` Bash
$ pip install requirements.txt
```

# Usage

```bash
$ cd project/code  
$ python main.py
```
Note : the default data path is `/data/test/`, the result is saved in `/data/result` as a `result.zip` file.

# Code Intro

This repo is mainly composed of three parts as following:
### Part1 Vehicle detection

Related code could be found in `project/train/efficientdet`.

Given a video file, EfficientDet is taken to output the size and position of vehicles via frame-by-frame.

Considering the trade-off between inference acurracy and speed, `efficientdet-d5` model is used,, see more in `weights/`.

Thanks a lot to the efforts from zylo117 [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch).

In `temp_data/`, the results are saved as json files.

### Part2 Lane detection 
Related code could be found in `project/train/PINet`.

PINet is taken to detect the lane points and output the results in `temp_data`.

Thanks a lot to the efforts from koyeongmin [PINet_new](https://github.com/koyeongmin/PINet_new).

### Part3 Cross detection
This is the most important part that is contributed to the final result. Related code could be found in `project/code/crossDet`.

Specifically, vehicle tracking, lane tracking ,lane type classfication, ROI mask, occlussion object are included in this part. See more in `utils.py`


# Docker
1. build image

```bash
docker build -t {image_name} .
```
2. load image
```Bash
docker load my_image.tar
nvidia-docker run --name {container_name} {my_image_name}
```
