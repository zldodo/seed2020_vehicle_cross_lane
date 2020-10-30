
import os,time,json
import sys
import cv2
import torch
sys.path.append('../train/efficientDet/')
from efficientdet_test_seed import main as main_vehicle_det
from efficientdet_test_seed import obj_list,anchor_ratios,anchor_scales
from backbone import EfficientDetBackbone

sys.path.append('../train/PINet/')
from test_seed import main as main_lane_det
from parameters import Parameters
import agent
p = Parameters()

from crossDet.main_video import main as main_cross_det
from crossDet.utils import save_json



data_dir = './../../../data/test2/'
res_dir = './../../../data/result2/'
test_idx = '1'
lane_dir = '../temp_data/' + test_idx + '/lane_json/'
vehicle_dir = '../temp_data/' + test_idx + '/vehicle_json/'
# img_dir = 'C:/D/002_DeepLearning/crossLane/demo/example/images/' + test_idx + '/'
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

start_time = time.clock()
# vehicle detection
compound_coef = 5
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                            ratios=anchor_ratios, scales=anchor_scales)
# load model file
model.load_state_dict(torch.load(f'../train/efficientDet/weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
model.eval()

# lane detection
print('Get agent')
lane_agent = agent.Agent()
lane_agent.load_weights(32, "tensor(1.1001)")

print('Setup GPU mode')
if torch.cuda.is_available():
    lane_agent.cuda()
lane_agent.evaluate_mode()

video_list = os.listdir(data_dir)

for each in video_list:
    video_name = each.split('.mp4')[0]
    # video_name = video_name[-2:]
    video_file = data_dir + each

    cap = cv2.VideoCapture(video_file)
    img_name = 0
    img_names = []
    res = []
    trackers =[]
    timestamps = []
    
    while(cap.isOpened()):
        ret, ori_frame = cap.read()
        if not ret:
            break
        frame_time = round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0,2)

        vehicle_data = main_vehicle_det(model,ori_frame,img_name,video_name)
        # print(vehicle_data)

        lanes = main_lane_det(lane_agent,ori_frame,img_name,video_name)
    #     img_names.append(img_name)
    #     img_name +=1
    #     timestamps.append(frame_time)
    # cap.release()
    # for img_name in img_names:
    #     with open(lane_dir+str(img_name)+'_lanes_info.json','r') as f:
    #         lanes= json.load(f)

    #     with open(vehicle_dir+str(img_name)+'_vehicle_info.json','r') as f:
    #         vehicle_data= json.load(f)

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

        # frame_time = timestamps[img_name]
        # ori_frame = cv2.imread(img_dir+str(img_name)+'.jpg')
        trackers,res, img_out = main_cross_det(ori_frame,vehicle_data,lanes_data,frame_time,trackers,res)
                
        # h,w,c = ori_frame.shape
        # cv2.namedWindow(f'video_{video_name}',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(f'video_{video_name}',int(w/2),int(h/2))
        # cv2.imshow(f'video_{video_name}',img_out)

        # display(img,lanes_data,vehicle_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_name +=1

    res_path = res_dir+str(video_name)+'.json'
    save_json(res,res_path)

    cap.release()
    cv2.destroyAllWindows()

# os.chdir('../../../data/')
# os.system('zip -r result.zip result/')
# os.system('rm -rf result/*')
# os.system('mv result.zip result/')

end_time = time.clock()
print('Running time: %s Seconds'%(end_time-start_time))#
