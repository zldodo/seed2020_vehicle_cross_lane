
import json
import cv2
from .utils import display,detect_vehicle_cross_line,draw_points,\
IsMatched,update_tracks,save_json,lane_tracking,fit_lane_points,\
draw_lane,remove_occluded_vehicle,match_vehicle
from .classify_lane_type import *
from .ekf_track import *
import time
import math

def main(ori_frame,vehicle_data,lanes_data,frame_time,trackers,pre_res):
    vehicle_data_filtered = remove_occluded_vehicle(vehicle_data)
    fitted_lane,meas_z =  fit_lane_points(lanes_data,ori_frame)
    lanes_type = classify_lane_type(ori_frame,fitted_lane)
    if not trackers:

        for z, style in zip(meas_z,lanes_type):
            # theta = math.atan(z[0])
            # X0 = np.array([[theta],[z[1]],[0],[0]])
            X0 = np.array([[z[0]],[z[1]],[0],[0]])
            tracker = init_lane_tracker(X0,style,500)
            trackers.append(tracker)
        # print('init lane tracks') 
    else:
        trackers = lane_tracking(trackers,lanes_data,meas_z,lanes_type)

    # img_lane = draw_lane(ori_frame,trackers)
    match_res = match_vehicle(vehicle_data,pre_res)
    out,cross,trackers = detect_vehicle_cross_line(ori_frame,vehicle_data_filtered,trackers,match_res)
    res = []
    for one in cross:
        t =  frame_time
        mx,my,w,h= one[0],one[1],one[2],one[3]
        lane_type = 1 if one[4]=='solid' else 0
        #print(one[4])
        res.append([t,mx,my,w,h,lane_type])

    # # fused with previous frames
    new_res = update_tracks(res,pre_res)
    # display(ori_frame,lanes_data,vehicle_data)

    return trackers, new_res, out
