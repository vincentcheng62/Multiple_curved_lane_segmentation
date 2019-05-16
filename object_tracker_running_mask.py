from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

# load weights and set defaults
config_path='config/yolov3-spp.cfg'
weights_path='config/yolov3-spp.weights'
class_path='config/coco.names'
img_size=608
conf_thres=0.2
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
#model.cuda()
model.eval()

classes = utils.load_classes(class_path)
#Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

#videopath = 'many_way.MOV'
#videopath = 'bridge.mp4'
videopath = 'lane_trim.mp4'

import cv2
from sort import *
#colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255),(255,255,255),(0,0,0),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

top_skip=230

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (1600,900))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
frame = frame[top_skip:1079, :]
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
mask = np.zeros(frame.shape, dtype=np.uint8)
result = np.zeros(frame.shape, dtype=np.uint8)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-lanedet.mp4"),fourcc,20.0,(vw,vh))

frames = 0
starttime = time.time()
obj_prev_markpt = {}
while(True):
    ret, frame = vid.read()

    #only act on the clear lane
    frame = frame[top_skip:1079, :]

    resultoverlayraw=frame.copy()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        print("one frame finished with ", len(tracked_objects))
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]

            cv2.rectangle(resultoverlayraw, (x1, y1), (x1+box_w, y1+box_h), (0,255,0), 2)
            cv2.rectangle(resultoverlayraw, (x1, y1-35), (x1+len(cls)*19+80, y1), (0,255,0), -1)
            cv2.putText(resultoverlayraw, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)


            markpt = (int(x1+box_w*0.5) ,y1+box_h)
            cv2.circle(mask, markpt, 3, 255, -1)

            if obj_id in obj_prev_markpt:
                cv2.line(mask, obj_prev_markpt[obj_id], markpt, 255, 1)

            obj_prev_markpt[obj_id]=markpt



    #cv2.imshow('Stream', frame)
    #

    #cv2.imwrite("maskmask.jpg", mask)
    mask_copy = mask.copy()
    mask_copy_gray = cv2.cvtColor(mask_copy,cv2.COLOR_BGR2GRAY)
    im1_new, contours_new, hierarchy_new = cv2.findContours(mask_copy_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    obj_id=0                                                                                                                                                                    
    for cnt in contours_new:
        #rect = cv.minAreaRect(cnt)
        #area = rect[1][0]*rect[1][1]
        #if area > 1200 and max(rect[1][0], rect[1][1])>350:
        color = colors[int(obj_id) % len(colors)]
        obj_id=obj_id+1
        cv2.drawContours(result, [cnt], 0, color, -1) 
        cv2.drawContours(resultoverlayraw, [cnt], 0, color, -1) 

    print("Total lane segment: ", obj_id)
    print("Frames: ", frames)
    cv2.imwrite("segment_result.jpg", result)
    cv2.imshow('Stream', resultoverlayraw)
    outvideo.write(resultoverlayraw)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27 or ch == 113:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
outvideo.release()
