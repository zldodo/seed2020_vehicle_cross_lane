import os,cv2
import numpy as np
from PIL import Image

def find_lane_pixels(img,lanes_data):
    x = lanes_data['x']
    y = lanes_data['y']
    
    mask = np.zeros_like(img)
    h,w = img.shape
    #print('img_size(h,w,c) = {},{},{}'.format(h,w,c))
    patch_size=15
    for k in range(len(x)):
        for i,j in zip(x[k],y[k]):
            low_x = 0 if i-patch_size<0 else i-patch_size 
            low_y = 0 if j-patch_size<0  else j-patch_size
            upper_x = w if i+patch_size>w else i+patch_size
            upper_y = h if j+patch_size>h else j+patch_size 
            #print('lane_area(x_l,y_l,x_u,y_u) = {},{},{},{}'.format(low_x,low_y,upper_x,upper_y))
            mask[low_y:upper_y,low_x:upper_x]=img[low_y:upper_y,low_x:upper_x]
    
    return mask 

def HLS_select(img, thresh=(120, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    l_channel = l_channel*(255/np.max(l_channel))
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output


def Sobel_select(img, orient='x', thresh_min=30, thresh_max=100):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

def Canny_select(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150)

    return edges

#def fit_select(img,lane_area):

def curve_fit(image,lanes_data):

    x_data = lanes_data['x']
    y_data = lanes_data['y']

    h,w,c = image.shape
    mask = np.zeros((h,w))
    mask_color = np.zeros((h,w))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lane_index = 0
    fit_x =[]
    fit_y = []
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
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed,元素变为pts
        patch_size=3
        for i,j in zip(draw_x,draw_y):
            low_x = 0 if i-patch_size<0 else i-patch_size 
            low_y = 0 if j-patch_size<0  else j-patch_size
            upper_x = w-1 if i+patch_size>w-1 else i+patch_size
            upper_y = h-1 if j+patch_size>h-1 else j+patch_size 
            #print('lane_area(x_l,y_l,x_u,y_u) = {},{},{},{}'.format(low_x,low_y,upper_x,upper_y))
            mask[low_y:upper_y,low_x:upper_x]=1
            # lane_gray=gray[low_y:upper_y,low_x:upper_x,1]
            
            # _,lane_type = classify_lane_type(lane_gray)
            # if lane_type==0:
            #     mask[low_y:upper_y,low_x:upper_x]=1
            # lanes_data['lane_'+str(lane_index)]['type']=lane_type
        lane_index = lane_index+1
        out = np.multiply(gray[:,:,1],mask)
        mask_color= cv2.inRange(out,120,255)
        #mask[draw_y,draw_x]=1

        cv2.polylines(image, [draw_points], False, (0,0,255), 5)
        fit_x.append(draw_x)
        fit_y.append(draw_y)
    fitted_points = {}
    fitted_points['x'] = fit_x
    fitted_points['y'] = fit_y
        
    return mask,mask_color,image, fitted_points,zs

def extract_dash_line(img):
    kernel1 = np.ones((3,5),np.uint8)
    kernel2 = np.ones((9,9),np.uint8)

    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBW=cv2.threshold(imgGray, 230, 255, cv2.THRESH_BINARY_INV)[1]

    img1=cv2.erode(imgBW, kernel1, iterations=1)
    img2=cv2.dilate(img1, kernel2, iterations=3)
    img3 = cv2.bitwise_and(imgBW,img2)
    img3= cv2.bitwise_not(img3)
    img4 = cv2.bitwise_and(imgBW,imgBW,mask=img3)
    imgLines= cv2.HoughLinesP(img4,15,np.pi/180,10, minLineLength = 8, maxLineGap = 4)

    for i in range(len(imgLines)):
        for x1,y1,x2,y2 in imgLines[i]:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    return img

def classify_lane_type(img,lanes_data):
    w,h,c= img.shape
    # 1) image preprocess
    # a)  weighted mean gray scale, b) mean filter and gray transformation
    # c) ROI and binary scale
    gray = img.copy()
    one_channel = img[:,:,0]*0.114 +  img[:,:,1]*0.587 + img[:,:,2]*0.299 # BGR
    # gray = np.expand_dims(gray,axis=2).astype(np.uint8)
    gray[:,:,0] =one_channel.astype(np.uint8)
    gray[:,:,1] =one_channel.astype(np.uint8)
    gray[:,:,2] =one_channel.astype(np.uint8)
    # print("gray image shape : {}, type :{}".format(gray.shape,type(gray)))
    #gray = np.array(gray,dtype='unit8')
    img_mean= cv2.medianBlur(gray,5)
    # img_mean = gray

    pixel_max = img_mean.max()
    pixel_min = img_mean.min()

    table=[]
    for i in range(256):
        table.append(200*(i-pixel_min)/(pixel_max-pixel_min))
    IMG = Image.fromarray(np.uint8(img_mean[:,:,0]))
    img_tansfer=IMG.point(table,'L')
    img_tansfer = np.asarray(img_tansfer)

    # reserve ROI area
    img_roi = find_lane_pixels(img_tansfer,lanes_data)
    mask= cv2.inRange(img_roi,120,255)

    res = []
    # 2) method 1 : white points is less than 0.75
    # method 2 : the maximum continuous black points is greater than 0.3 
    
    for x,y in zip(lanes_data['x'],lanes_data['y']):
        
        lane = np.zeros((len(x),1))
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]

            if mask[yi,xi]>0: lane[i] = 1
            elif xi-1 >= 0 and mask[yi,xi-1]>0: lane[i] = 1
            elif xi-2 >= 0 and mask[yi,xi-2]>0: lane[i] = 1
            elif xi-3 >= 0 and mask[yi,xi-3]>0: lane[i] = 1
            elif xi-4 >= 0 and mask[yi,xi-4]>0: lane[i] = 1
            elif xi-5 >= 0 and mask[yi,xi-5]>0: lane[i] = 1
            elif xi-6 >= 0 and mask[yi,xi-6]>0: lane[i] = 1
            elif xi+1 <= w-1 and mask[yi,xi+1]>0: lane[i] = 1
            elif xi+2 <= w-1 and mask[yi,xi+2]>0: lane[i] = 1
            elif xi+3 <= w-1 and mask[yi,xi+2]>0: lane[i] = 1
            elif xi+4 <= w-1 and mask[yi,xi+4]>0: lane[i] = 1
            elif xi+5 <= w-1 and mask[yi,xi+5]>0: lane[i] = 1
            elif xi+6 <= w-1 and mask[yi,xi+6]>0: lane[i] = 1
    
        # TODO : delete those points covered by vehicle
        dash_1 = True if sum(lane)<len(lane)*0.75 else False
        count = 0
        max_count = 0
        for i in range(len(lane)):
            if lane[i] == 0:
                count +=1
                if count>max_count:
                    max_count = count
            else: 
                count = 0

        dash_2 =  True if max_count>len(lane)*0.5 else False

        dash = dash_1 and dash_2
        text = 'dash' if dash else 'solid'
        res.append(text)

    return res