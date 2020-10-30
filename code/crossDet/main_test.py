
import json
import cv2
from utils import display,detect_vehicle_cross_line,draw_points,\
IsMatched,update_tracks,save_json,lane_tracking
from classify_lane_type import *
import time

test_idx = '2'
img_dir = 'C:/D/002_DeepLearning/crossLane/demo/example/images/' + test_idx + '/'
lane_dir = 'C:/D/002_DeepLearning/PINet_new-master/CurveLanes/res_json/' + test_idx + '/'
vehicle_dir = 'C:/D/002_DeepLearning/efficientDet/Yet-Another-EfficientDet-Pytorch-master/res/' + test_idx + '/'
file_index = 10
img_list = os.listdir(img_dir)
lane_tracks = {}
lane_tracks['x']=[]
lane_tracks['y']=[]
polys = []
pre_res =[]
# for file_index in range(len(img_list)):
with open(lane_dir+str(file_index)+'_lanes_info.json','r') as f:
    lanes= json.load(f)
    #print(lane_data['lane_0'])
    lane_index = 0
    x = []
    y = []
    lanes_data ={}
    while lanes.get('lane_'+str(lane_index)) is not None:
        lane = lanes.get('lane_'+str(lane_index))
        #print(lane)
        x.append(lane['x'])
        y.append(lane['y'])
        lane_index = lane_index+1
    lanes_data['x'] = x
    lanes_data['y'] = y

with open(vehicle_dir+str(file_index)+'_vehicle_info.json','r') as f:
    vehicle_data= json.load(f)
    #print(vehicle_data['rois'])

img = cv2.imread(img_dir+str(file_index)+'.jpg')

if lane_tracks is None:
    lane_tracks = lanes_data
else:
    lane_tracks = lane_tracking(lanes_data,polys,lane_tracks)
lane_points,mask_color,img_lane,fitted_lane,polys= curve_fit(img,lane_tracks)

lanes_type = classify_lane_type(img,fitted_lane)
num = 0
for i in range(len(lanes_type)):
    cv2.putText(img_lane, '{}:{}'.format(num,lanes_type[i]),\
    (fitted_lane['x'][i][0], fitted_lane['y'][i][0] - 2),0,3.0/3,\
        [0, 0, 0],thickness=3, lineType=cv2.FONT_HERSHEY_SIMPLEX)
    num+=1


# img = draw_points(img,mask_color)
out,cross = detect_vehicle_cross_line(img,vehicle_data,lanes_data)
res = []
for one in cross:
    t =  file_index
    mx,my,w,h= one[0],one[1],one[2],one[3]
    lane_type = 1 if lanes_type[one[4]]=='solid' else 0
    print(one[4],lanes_type[one[4]])
    res.append([t,mx,my,w,h,lane_type])

# fused with previous frames
pre_res = update_tracks(res,pre_res)
print("res base:")
print(pre_res)
        
h,w,c = img.shape
# cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('mask',int(w/2),int(h/2))
# cv2.imshow('mask',mask_color)

cv2.namedWindow('out',cv2.WINDOW_NORMAL)
cv2.resizeWindow('out',int(w/2),int(h/2))
cv2.imshow('out',img_lane)

display(img,lanes_data,vehicle_data)
cv2.waitKey(0)
# print(file_index)
# time.sleep(0.5)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break