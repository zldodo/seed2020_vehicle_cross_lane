import os
import cv2
import numpy as np
from matplotlib import path
import json
import math
from .ekf_track import *

color_list = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]
def draw_points(img,lanes_data):

    x = lanes_data['x']
    y = lanes_data['y']
    # print('the num of lanes : {}'.format(len(x)))
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            img = cv2.circle(img, (int(i[index]), int(j[index])), 5, color_list[color_index], -1)
    return img

def plot_one_box(img, coord, score=None, color=(255,0,0), line_thickness=None):
    if score > 0.5:
        tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)

def display(img,lanes_data,vehicle_data):
    h,w,c = img.shape
    img = draw_points(img,lanes_data)
    
    for j in range(len(vehicle_data['rois'])):
            x1, y1, x2, y2 = vehicle_data['rois'][j]
            score = float(vehicle_data['scores'][j])
            plot_one_box(img, [x1, y1, x2, y2],score=score,color=color_list[1])
    vertices = np.array([[0.4*w, 0], [0.4*w, 0.3*h],[0,0.5*h],[0,h],\
                    [w,h],[w,0.5*h],[0.6*w,0.3*h],[0.6*w,0]],np.int)
    cv2.polylines(img, [vertices], 1,(0,255,0),2)
    left_vertices = np.array([[0,0],[0.4*w, 0], [0.4*w, 0.4*h],[0,h]],np.int)
    right_vertices = np.array([[w,h],[w,0],[0.6*w,0],[0.6*w,0.4*h]],np.int)
    cv2.polylines(img, [left_vertices], 1,(255,0,0),2)
    cv2.polylines(img, [right_vertices], 1,(255,0,0),2)
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame',int(w/2),int(h/2))
    cv2.imshow('frame',img)

def draw_lane(img,trackers):
    # print('there are {} lanes in tracking'.format(len(trackers)))
    img_out = img.copy()
    for tracker in trackers:
        h,w,c = img.shape
        k = tracker.X[0,0]
        # k = math.tan(tracker.X[0,0])
        b = tracker.X[1,0]
        z = np.array([k,b])
        # print('z is {}'.format(z))
        start = h*0.4
        end = h-1
        # start = h*0.5 if min(y_list)>h*0.5 and len(y_list)>15 else min(y_list)
        # # end   = max(max(y_list), image.shape[0] / 2)
        # # end = max(y_list)
        # end = h-1 if max(y_list)> h/2+100 and min(x_list)>200 and max(x_list)<w-200 and len(x_list) > 8 else max(y_list)
        # end = h-1 if end > h else end
        lspace = np.linspace(start, end, 300) # 等间距采点,限制了输出高度、即也限制了输出距离

        draw_y = lspace.astype(np.int32)
        # print('draw_y shape is {}'.format(draw_y.shape))
        draw_x = np.polyval(z, draw_y).astype(np.int32)   # evaluate the polynomial，自变量draw_x，y不一定单调
        # print('draw_x shape is {}'.format(draw_x.shape))
        draw_x = [i if i>=0 else 0 for i in draw_x]
        draw_x = [i if i<w else w-1 for i in draw_x]
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed,元素变为pts
        
        cv2.polylines(img_out, [draw_points], False, (0,0,255), 5)
        latest_style= tracker.latest_style
        history_style = 'solid' if tracker.style_count[0] > tracker.style_count[1]/3\
                                else 'dash'
        cv2.putText(img_out, 's{}: d{}: c{}'.format(tracker.style_count[0],tracker.style_count[1],\
        tracker.cross_count),(draw_x[0], draw_y[0] - 2),0,3.0/3,\
        [0, 0, 0],thickness=3, lineType=cv2.FONT_HERSHEY_SIMPLEX)
        # cv2.putText(img_out, '{}:{}'.format(latest_style,history_style),\
        # (draw_x[0], draw_y[0] - 2),0,3.0/3,\
        # [0, 0, 0],thickness=3, lineType=cv2.FONT_HERSHEY_SIMPLEX)
    return img_out
        

def detect_vehicle_cross_line(img,vehicle_data,trackers,match_res):
    h,w,c = img.shape
    # fit_x,fit_y,_ = fit_lane_points(lanes_data,img)
    results =[]
    for j in range(len(vehicle_data['rois'])):
        x1, y1, x2, y2 = vehicle_data['rois'][j]
        score = float(vehicle_data['scores'][j])
        v_w = x2-x1 # vehicle width
        v_h = y2-y1 # vehicle height

        # select the focus area of image since the edge is not concerned
        middle_x = (x1 + x2)/2.0 
        middle_y = (y1 + y2)/2.0 
        img = cv2.circle(img, (int(middle_x),int(middle_y)), 5, (255,0,0), -1) 

        # # filtered by ROI area
        # select =  True if isInROI(middle_x,middle_y,w,h) and (v_w >80 and v_h>80)else False
        # if not select:
        #     # print("({},{})not in ROI!".format(middle_x,middle_y))
        #     continue
        
        # get the pos of vehicle in the image 
        pos = isOnLeftOrRight(middle_x,middle_y,w,h)

        # filtered out those small vehicles on the side of image
        if pos!=1 and (v_w <100 or v_h<85): continue 

        # judge if the vehicle is crossing the lane
        # # 1) find the correspoind xi of lane point with y=y2
        # # 2) if x1<xi<x2, then vehicle is crossing they lane
        
        # lane_idx = 0
        # for i in range(len(fit_x)):
        #     points_x = fit_x[i]
        #     points_y = fit_y[i]
        for tracker in trackers:
            
            # filtered by ROI area
            select =  True if isInROI(middle_x,middle_y,w,h) and (v_w >80 and v_h>80)else False
            if not select and not j in match_res :
                # print("({},{})not in ROI!".format(middle_x,middle_y))
                continue
            # k,b =math.tan(tracker.X[0]),tracker.X[1]
            # z = np.array([tracker.X[0,0],tracker.X[1,0]])
            # _y = lspace.astype(np.int32)
            # if y2 < tracker.start - 100 and y2 < 0.6 *h:
            #     continue 
            # cross_x = np.polyval(z, y2)
            k = tracker.X[0,0]
            # k = math.tan(tracker.X[0,0])
            b = tracker.X[1,0]
            cross_x = k*y2+b
            # print('cross_x,cross_x2 = {},{}'.format(cross_x,cross_x2))
            # idx,_ = find_nearest(points_y,y2)
            # if idx==None: continue
            # cross_x = points_x[idx]
            # on the side of image : v_w/v_h > 1.8/1.4, the ratio of lane is close to x
            # left or right side
            # r = 1.2
            # if v_w/v_h > 1.8/1.4*r:
            # TODO needs to be optimized
            # if pos == 0: res = 1 if cross_x > 10 and cross_x < x2 - v_w*0.5 and cross_x>x1 else 0
            # elif pos ==2 :res = 1 if cross_x < w-10 and cross_x > x1 + v_w*0.5 and cross_x <x2 else 0
            # else : res = 1 if cross_x > x1 and cross_x > x1-20 and cross_x < x2+20 else 0
            # when the lane is almost on the left/right of object obx
            # when the vehicle is oncoming

            # where is the wheel
            res = 0
            if cross_x > x1 -20 and cross_x < x2+20:
                if pos ==1: res = 1
                elif pos ==0:
                    res =1 if cross_x < x2 - v_w*0.5 and cross_x > 0 else 0
                else:
                    res =1 if cross_x < w and cross_x > x1 + v_w*0.5 else 0

            if res == 1:
                # if tracker.ttl > 2:
                lane_type = tracker.get_lane_style()
                results.append([int(middle_x),int(middle_y),int(v_w),int(v_h),lane_type])
                plot_one_box(img, [x1, y1, x2, y2],score=score,color=color_list[1])
                if sum(tracker.style_count) > 5:
                    tracker.cross_count += 1    
                break
            # lane_idx += 1
    return img,results,trackers

def fit_lane_points(lanes_data,img):
    h,w,c = img.shape
    x_data = lanes_data['x']
    y_data = lanes_data['y']
    
    fit_x =[]
    fit_y =[]
    fit_lines = []
    zs = []
    for x_list,y_list in zip(x_data,y_data):
        x = np.array(x_list).astype(np.float32)
        y = np.array(y_list).astype(np.float32)
        #calculate the coefficients.需要调节选用的曲线次数
        z = np.polyfit(y, x, 1) # z为多项式系数矩阵，从高次到低次
        zs.append(z)

        start = h*0.5 if min(y_list)>h*0.5 and len(y_list)>15 else min(y_list)
        # end   = max(max(y_list), image.shape[0] / 2)
        # end = max(y_list)
        end = h-1 if max(y_list)> h/2+100 and min(x_list)>200 and max(x_list)<w-200 and len(x_list) > 8 else max(y_list)
        end = h-1 if end > h else end
        lspace = np.linspace(start, end, 300) # 等间距采点,限制了输出高度、即也限制了输出距离

        draw_y = lspace.astype(np.int32)
        draw_x = np.polyval(z, draw_y).astype(np.int32)   # evaluate the polynomial，自变量draw_x，y不一定单调
        draw_x = [i if i>=0 else 0 for i in draw_x]
        draw_x = [i if i<w else w-1 for i in draw_x]
        # draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed,元素变为pts

        # cv2.polylines(image, [draw_points], False, (0,0,255), 5)
        fit_x.append(draw_x)
        fit_y.append(draw_y)
        fit_lines.append(z)
    fitted_points = {}
    fitted_points['x'] = fit_x
    fitted_points['y'] = fit_y
        
    
    return fitted_points,fit_lines


def find_nearest(data,value):
    data = np.asarray(data)
    idx = (np.abs(data - value)).argmin()
    if abs(value - data[idx]) < 20:
        return idx, data[idx]
    else: 
        return None,None

def point_distance_line(py,px,poly):
    dec = math.fabs(np.polyval(poly,py)-px)
    num = math.pow(poly[0]**2+poly[1]**2,0.5)
    dist = dec/num
    return dist

def lane_tracking(trackers,lane_meas,meas_z,lanes_type,thresh = 0.5):

    if lane_meas is None: return trackers
    nMeas = len(lane_meas)
    nTrack = len(trackers)
    matched_polys = np.zeros(nTrack)
    ref_trackers = trackers[:nTrack]
    for xs,ys,z,style in zip(lane_meas['x'],lane_meas['y'],meas_z,lanes_type):
        addNew = True
        dist_count = []
        for i, tracker in enumerate(ref_trackers):
            # k = math.tan(tracker.X[0,0])
            k = tracker.X[0,0]
            poly = np.array([k,tracker.X[1,0]])
            nInline = 0
            
            dist =[point_distance_line(y,x,poly) for x,y in zip(xs,ys)]
            dist_count.append(dist)

        # association
        sum_dist = np.sum(dist_count,axis =1)
        inlines = np.array(dist_count) < thresh
        nInline = np.sum(inlines,axis =1)
        min_dist = max(sum_dist)
        for i,count in enumerate(nInline):
            if count > len(xs) * 0.6:
                if sum_dist[i] < min_dist:
                    min_dist = sum_dist[i] 
                    addNew = False
                    id_match = i
        
        if addNew:
                # add an new lane track
                # print('add an new lane track')
                # theta = math.atan(z[0])
                # meas = np.array([[theta],[z[1]],[0],[0]])
                meas = np.array([[z[0]],[z[1]],[0],[0]])
                tracker = init_lane_tracker(meas,style,ys[0])
                trackers.append(tracker)
        else:
            meas = np.array([[z[0]],[z[1]]])
            trackers[id_match].update(meas,style,ys[0])
            matched_polys[id_match] += 1

    # print('list the matched polys:')
    # print(matched_polys)
    # update lane tracks
    
    for i,c in enumerate(matched_polys):
        if c == 0.:
            trackers[i].ttl -=1
            trackers[i].predict()
    # print('list the ttl of lanes:')
    # ttls = [tracker.ttl for tracker in trackers]
    # print(ttls)
    # print('list the update count of lanes:')
    # updates = [tracker.style_count for tracker in trackers]
    # print(updates)
    
    trackers = [tracker for tracker in trackers \
                    if (tracker.ttl >-10 and tracker.pred_count<10)\
                        or ( tracker.pred_count<20 and tracker.cross_count > 5)
    ] 
    
    return trackers

def isInROI(x,y,w,h):
    
    vertices = np.array([[0.4*w, 0], [0.4*w, 0.3*h],[0,0.5*h],[0,h],\
                    [w,h],[w,0.5*h],[0.6*w,0.3*h],[0.6*w,0]])
    roi = path.Path(vertices)
    return roi.contains_points([(x,y)])

def isOnLeftOrRight(x,y,w,h):
    left_vertices = np.array([[0,0],[0.4*w, 0], [0.4*w, 0.4*h],[0,h]])
    right_vertices = np.array([[w,h],[w,0],[0.6*w,0],[0.6*w,0.4*h]])
    left_roi = path.Path(left_vertices)
    right_roi = path.Path(right_vertices)
    if left_roi.contains_points([(x,y)]):
        pos = 0
    elif right_roi.contains_points([(x,y)]):
        pos = 2
    else: pos = 1
    return pos

def IsMatched(rec1,rec2):
    x1,y1,x2,y2 = rec1[1]-rec1[3]/2.0,rec1[2]-rec1[4]/2.0,rec1[1]+rec1[3]/2.0,rec1[2]+rec1[4]/2.0
    x3,y3,x4,y4 = rec2[1]-rec2[3]/2.0,rec2[2]-rec2[4]/2.0,rec2[1]+rec2[3]/2.0,rec2[2]+rec2[4]/2.0
    # print('rec1:{},{},{},{}'.format(x1,y1,x2,y2))
    # print('rec2:{},{},{},{}'.format(x3,y3,x4,y4))
    if (x2<=x3 or x4<=x1) and (y2 <= y3 or y4<=y1):
        return False
    else:
        lens = min(x2, x4) - max(x1, x3)
        wide = min(y2, y4) - max(y1, y3)
    iou = (lens*wide)
    s1 = (x2-x1)*(y2-y1)
    s2 = (x4-x3)*(y4-y3)
    if iou/float(s1) > 0.6 and iou/float(s2) > 0.6:
        return True
    else:
        return False

def update_tracks(meas,tracks):
    # if not meas:
        
        # print("no measuremnt is found")
    # else:
    if meas:
        # print("current res:")
        # print(res)
        new_track = []
        if not tracks:
            for each in meas:
                # print('init the container successfully')
                tracks.append([each])
        else:
            for each in meas:
                # find the matched vehicle in previous frames
                addNew = True
                for track in tracks: 
                    for i in range(min(len(track),12)):
                        pre_object = track[-(i+1)]
                        # print(pre_object)
                        t_pre = pre_object[0]
                        t_cur = each[0]
                        iouMatch = IsMatched(pre_object,each)
                        if iouMatch and t_cur-t_pre <8:
                            addNew = False
                            # print('add object({},{}) into the track({},{})'.format\
                            # (each[1],each[2],pre_object[1],pre_object[2]))
                            track.append(each)
                            break
                    if not addNew : break
                if addNew:
                    # print('add an new track: {},{},{},{}'.format(each[1],each[2],each[3],each[4]))
                    new_track.append(each)
            if new_track is not None:
                for each in new_track:
                    tracks.append([each])  

    return tracks

def save_json(res,res_path):
    results =[]
  
    for each in res:
        if len(each)<12: continue
        else:
            start_time = each[0][0]
            xs,ys = each[0][1],each[0][2]
            ws,hs = each[0][3],each[0][4]

            end_time = each[-1][0]
            xe,ye = each[-1][1],each[-1][2]
            we,he = each[-1][3],each[-1][4]

            # remove small obejcts
            if max(ws,we) < 90 or max(hs,he) <90 : continue

            dash,solid =0,0
            for i in each:
                if i[5] == 0: dash+=1
                else: solid+=1

            lane_type = 'dash' if dash>solid*2 else 'solid'
            item = {}
            item["start_time"] = start_time
            item["xs"], item["ys"]= xs,ys
            item["ws"], item["hs"]= ws,hs
            item["end_time"] = end_time
            item["xe"], item["ye"]= xe,ye
            item["we"], item["he"]= we,he
            item["line_style"]=lane_type
            # print(each)
            results.append(item)
    json_str=json.dumps(results)
    with open(res_path,'w') as f:
        f.write(json_str)


def save_only_one_json(res,times,res_path):
    results =[]

    most_item = 0
    for i, each in enumerate(res):
        if len(each) > most_item: 
            out_idx = i
            most_item = len(each)

    if most_item > 0:
        each  = res[out_idx]
        start_time = times[each[0][0]]
        xs,ys = each[0][1],each[0][2]
        ws,hs = each[0][3],each[0][4]

        end_time = times[each[-1][0]]
        xe,ye = each[-1][1],each[-1][2]
        we,he = each[-1][3],each[-1][4]

        dash,solid =0,0
        for i in each:
            if i[5] == 0: dash+=1
            else: solid+=1

        lane_type = 'dash' if dash>solid*2 else 'solid'
        item = {}
        item["start_time"] = start_time
        item["xs"], item["ys"]= xs,ys
        item["ws"], item["hs"]= ws,hs
        item["end_time"] = end_time
        item["xe"], item["ye"]= xe,ye
        item["we"], item["he"]= we,he
        item["line_style"]=lane_type
        results.append(item)
    json_str=json.dumps(results)
    with open(res_path,'w') as f:
        f.write(json_str)

def save_json_time(res,times,res_path):
    results =[]
  
    for each in res:
        if len(each)<12: continue
        else:
            start_time = times[each[0][0]]
            xs,ys = each[0][1],each[0][2]
            ws,hs = each[0][3],each[0][4]

            end_time = times[each[-1][0]]
            xe,ye = each[-1][1],each[-1][2]
            we,he = each[-1][3],each[-1][4]

            # remove small obejcts
            if max(ws,we) < 80 or max(hs,he) <80 : continue

            dash,solid =0,0
            for i in each:
                if i[5] == 0: dash+=1
                else: solid+=1

            lane_type = 'dash' if dash>solid*2 else 'solid'
            item = {}
            item["start_time"] = start_time
            item["xs"], item["ys"]= xs,ys
            item["ws"], item["hs"]= ws,hs
            item["end_time"] = end_time
            item["xe"], item["ye"]= xe,ye
            item["we"], item["he"]= we,he
            item["line_style"]=lane_type
            results.append(item)
            # print(each)
    json_str=json.dumps(results)
    with open(res_path,'w') as f:
        f.write(json_str)

def remove_occluded_vehicle(vehicle_data):
    remove_ids = []
    for j in range(len(vehicle_data['rois'])):
        x1, y1, x2, y2 = vehicle_data['rois'][j]
        for i in range(len(vehicle_data['rois'])):
            if i==j : continue
            if i in remove_ids : continue
            if j in remove_ids : continue
            x3, y3, x4, y4 = vehicle_data['rois'][i]
            if (x2<=x3 or x4<=x1) and (y2 <= y3 or y4<=y1):
                continue
            else:
                lens = min(x2, x4) - max(x1, x3)
                wide = min(y2, y4) - max(y1, y3)
            iou = (lens*wide)
            s1 = (x2-x1)*(y2-y1)
            s2 = (x4-x3)*(y4-y3)
            if float(s1)==0.0 or float(s2)==0.0 : continue
            if iou/float(s1) > 0.3 or iou/float(s2) > 0.3 \
                or lens>0.8*abs(x1-x2) or lens>0.8*abs(x3-x4):
                if y2 > y4 :
                    remove_ids.append(i)
                else:
                    remove_ids.append(j)
    res = {}
    res['rois'] =[]
    res['scores'] =[]
    for j in range(len(vehicle_data['rois'])):
        if j not in remove_ids:
            # print('add {}th vehicle'.format(j))
            res['rois'].append(vehicle_data['rois'][j])
            res['scores'].append(vehicle_data['scores'][j])
    
    # print(remove_ids)
    return res

def match_vehicle(vehicle_data,tracks):
    match_res=[]
    for j in range(len(vehicle_data['rois'])):
        x1, y1, x2, y2 = vehicle_data['rois'][j]
        v_w = x2-x1 # vehicle width
        v_h = y2-y1 # vehicle height
        middle_x = (x1 + x2)/2.0 
        middle_y = (y1 + y2)/2.0 
        each = [0,int(middle_x),int(middle_y),int(v_w),int(v_h),0]
        iouMatch = False
        for track in tracks:
            for i in range(min(2,len(track))):
                pre_object = track[-(i+1)]
                iouMatch = IsMatched(pre_object,each)
                if iouMatch and len(track)>10:
                    # match_res['j']= len(track)-i-1
                    match_res.append(j)
                    break
            if iouMatch:
                break
    return match_res

        
