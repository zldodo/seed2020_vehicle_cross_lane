"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2,os
import numpy as np
import json

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess,preprocess_one, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box



def display(pred, img, img_name,save_path, imshow=True, imwrite=False):
    
    if len(pred[0]['rois']) == 0:
        return

    img_vehicle = img[0].copy()

    for j in range(len(pred[0]['rois'])):
        x1, y1, x2, y2 = pred[0]['rois'][j].astype(np.int)
        obj = obj_list[pred[0]['class_ids'][j]]
        score = float(pred[0]['scores'][j])
        plot_one_box(img_vehicle, [x1, y1, x2, y2], label=obj,score=score,\
        color=color_list[get_index_label(obj, obj_list)])


    if imshow:
        cv2.imshow('img_vehicleDet', img_vehicle)
        cv2.waitKey(0)

    if imwrite:
        cv2.imwrite(f'{save_path}/img_inferred_d{compound_coef}_{img_name}.jpg',\
         img_vehicle)

############################################################################
## write result
############################################################################
def write_result_json(res_data,filename,save_path):
    out = res_data[0]
    res = {}
    rois = []
    scores = []
    for j in range(len(out['rois'])):
        obj = obj_list[out['class_ids'][j]]
        if obj not in vehicle_list:
            continue
        x1, y1, x2, y2 = out['rois'][j]
        score = float(out['scores'][j])
        rois.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(score)
    res['rois'] = rois
    res['scores'] = scores

    json_str = json.dumps(res)
    #print(save_path + str(filename) + '_vehicle_info.json')
    with open(save_path + str(filename) + '_vehicle_info.json','w') as f:
        f.write(json_str)
    
    return res


compound_coef = 5
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

vehicle_list = ['car','truck','bus']

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
save_root = './../temp_data/'

def main(model,img,img_name,video_name):
    
    ori_imgs, framed_imgs, framed_metas = preprocess([img],max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    save_json_path = save_root + str(video_name) + '/' + 'vehicle_json/'
    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path)
    res = write_result_json(out,img_name,save_json_path)

    save_infer_img_path = save_root + str(video_name) + '/' + 'img_infer_vehicle'
    if not os.path.exists(save_infer_img_path):
        os.makedirs(save_infer_img_path)
    # display(out, ori_imgs,img_name,save_infer_img_path,imshow=False, imwrite=False)
    
    return res

