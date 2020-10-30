import cv2,os
import json


video_dir = 'C:/D/002_DeepLearning/crossLane/first/test/data/video/'
# video_dir = 'C:/D/002_DeepLearning/crossLane/demo/example/videos/'
# save_path = video_path.split('.')[0]
save_dir = 'C:/D/002_DeepLearning/crossLane/first/test/data/timestamps/'
# save_dir = 'C:/D/002_DeepLearning/crossLane/demo/example/timestamps/'


video_list = os.listdir(video_dir)
video_list.sort()
print(video_list)
for video_name in video_list:
    save_path = save_dir+video_name.split('.mp4')[0]+'_timestamps.json'
    cap = cv2.VideoCapture(video_dir+video_name)
    i = 0
    timestamps = []
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        timestamps.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0,2))
        json_str = json.dumps(timestamps)
    print(save_path)
    print(len(timestamps))
    with open(save_path,'w') as f:
        f.write(json_str)
    cap.release()
    cv2.destroyAllWindows()


        

    