import cv2,os


# video_dir = 'C:/D/002_DeepLearning/crossLane/first/test/data/video/'
video_dir = 'C:/D/002_DeepLearning/crossLane/demo/example/videos/'
# save_path = video_path.split('.')[0]
save_dir = 'C:/D/002_DeepLearning/crossLane/first/test/data/image/'


video_list = os.listdir(video_dir)
video_list.sort()
print(video_list)
for video_name in video_list[:1]:
    # save_path = save_dir+video_name.split('.mp4')[0]
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # else:
    #     continue
    cap = cv2.VideoCapture(video_dir+video_name)
    i = 0
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'): 
        #     break

        # cv2.imwrite(save_path+'\%d.jpg'%i,frame)
        # i = i+1
        # if(cv2.waitKey(1)&0xFF) == ord('q'):
        #     break
    cap.release()
    cv2.destroyAllWindows()

    # print(timestamps[30],timestamps[98])
