import os

root = 'C:/D/002_DeepLearning/efficientDet/Yet-Another-EfficientDet-Pytorch-master/test/videos/'
subset = 'demo0'

image_path = root + subset
with open(subset+'.txt','w') as f:
    for p,d,fs in os.walk(image_path):
        for i in fs:
            image_addr = os.path.join(p,i)
            f.write(image_addr+'\n')

