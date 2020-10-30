
import json
import cv2
from utils import display,detect_vehicle_cross_line,draw_points,\
IsMatched,update_tracks,save_json,lane_tracking,fit_lane_points,\
draw_lane,save_json_time,remove_occluded_vehicle,match_vehicle
from classify_lane_type import *
from ekf_track import *
import time


test_idx = '11'
img_dir = 'C:/D/002_DeepLearning/crossLane/demo/example/images/' + test_idx + '/'
lane_dir = 'C:/D/002_DeepLearning/PINet_new-master/CurveLanes/res_json/' + test_idx + '/'
times_dir = 'C:/D/002_DeepLearning/crossLane/demo/example/timestamps/'+ test_idx + '_timestamps.json'
vehicle_dir = 'C:/D/002_DeepLearning/efficientDet/Yet-Another-EfficientDet-Pytorch-master/res/' + test_idx + '/'
res_dir = 'C:/D/002_DeepLearning/crossLane/demo/example/preds/'
# file_index = 71
img_list = os.listdir(img_dir)
print(len(img_list))
lane_tracks = {}
lane_tracks['x']=[]
lane_tracks['y']=[]
lane_tracks['ttl']=[]
polys = []
pre_res =[]
trackers =[]
with open(times_dir,'r') as f:
    timestamps= json.load(f)
for file_index in range(len(img_list)):
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
    
    vehicle_data_filtered = remove_occluded_vehicle(vehicle_data)

    img = cv2.imread(img_dir+str(file_index)+'.jpg')
    img = draw_points(img,lanes_data)

    fitted_lane,meas_z =  fit_lane_points(lanes_data,img)
    lanes_type = classify_lane_type(img,fitted_lane)
    if not trackers:

        for z, style in zip(meas_z,lanes_type):
            # theta = math.atan(z[0])
            # X0 = np.array([[theta],[z[1]],[0],[0]])
            X0 = np.array([[z[0]],[z[1]],[0],[0]])
            tracker = init_lane_tracker(X0,style,500)
            trackers.append(tracker)
        print('init lane tracks') 
    else:
        trackers = lane_tracking(trackers,lanes_data,meas_z,lanes_type)

    img_lane = draw_lane(img,trackers)
    match_res = match_vehicle(vehicle_data,pre_res)
    out,cross,trackers = detect_vehicle_cross_line(img_lane,vehicle_data_filtered,trackers,match_res)
    res = []
    for one in cross:
        t =  file_index
        mx,my,w,h= one[0],one[1],one[2],one[3]
        lane_type = 1 if one[4]=='solid' else 0
        print(one[4])
        res.append([t,mx,my,w,h,lane_type])
    
    # # fused with previous frames
    pre_res = update_tracks(res,pre_res)
            
    
    # cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('mask',int(w/2),int(h/2))
    # cv2.imshow('mask',mask_color)
    h,w,c = img.shape
    cv2.namedWindow('out',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('out',int(w/2),int(h/2))
    cv2.imshow('out',out)

    display(img,lanes_data,vehicle_data_filtered)
    time.sleep(0.1)
    # cv2.waitKey(0)
    print(file_index)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

res_path = res_dir+str(test_idx)+'.json'
save_json_time(pre_res,timestamps,res_path)