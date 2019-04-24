import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import imutils

from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
import random

from PIL import Image

# load weights and set defaults
config_path='config/yolov3-spp.cfg'
weights_path='config/yolov3-spp.weights'
class_path='config/coco.names'
img_size=608
conf_thres=0.05
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
#model.cuda()
model.eval()

classes = utils.load_classes(class_path)
#Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor

def linelength(line):
    return math.sqrt(math.pow((line[2]-line[0]),2)+math.pow((line[3]-line[1]),2))

def slope_in_angle(line):
    return math.degrees(math.atan2(1.0*line[3]-1.0*line[1], (1.0*line[2]-1.0*line[0])))

def slope(line):
    return (1.0*line[3]-1.0*line[1])/(1.0*line[2]-1.0*line[0])

def intersect(line1, line2): 
    pt1 = (line1[0], line1[1])
    pt2 = (line1[2], line1[3])
    ptA = (line2[0], line2[1])
    ptB = (line2[2], line2[3])
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
        
        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    dx1 = dx1 + random.random()*0.001
    dy1 = dy1 + random.random()*0.001

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;

    dx = dx + random.random()*0.001
    dy = dy + random.random()*0.001
    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0
    return ( xi, yi, 1, r, s )


def lines_merging(lines):
    new_extended_lines=[]
    lines_with_info=[]
    line_min_length=3
    lines_angle_diff_in_degree=12

    # make larger, something may miss the one and connect between (1) (-) (3)
    extension_ratio=5.0
    maxgap=300

    for line in lines:
        length=linelength(line[0])

        if length>line_min_length:
            lines_with_info.append([line[0], slope_in_angle(line[0]), length])

    for line1 in lines_with_info:
        for line2 in lines_with_info:

            abs_diff_in_angle = math.fabs(line1[1]-line2[1])
            if abs_diff_in_angle < lines_angle_diff_in_degree:

                (xi, yi, valid, r, s) = intersect(line1[0], line2[0])
                if valid == 1:

                    # r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
                    # s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
                    if r > 0: # intersection is along line1 original direction
                        if r >=1 and r < 1.0+extension_ratio and (r-1)*line1[2]< maxgap:
                            if s > 0 :
                                if s >=1 and s < 1.0+extension_ratio and (s-1)*line2[2]< maxgap:                                
                                    new_extended_lines.append([line1[0][0], line1[0][1], xi, yi])
                                    new_extended_lines.append([line2[0][0], line2[0][1], xi, yi])

                            else:
                                if s >= -1*extension_ratio and (-s)*line2[2]< maxgap:
                                    new_extended_lines.append([line1[0][0], line1[0][1], xi, yi])
                                    new_extended_lines.append([line2[0][2], line2[0][3], xi, yi])                                    
                    
                    else:
                        if r >= -1*extension_ratio and (-r)*line1[2]< maxgap:
                            if s > 0 :
                                if s >=1 and s < 1.0+extension_ratio and (s-1)*line2[2]< maxgap:                                
                                    new_extended_lines.append([line1[0][2], line1[0][3], xi, yi])
                                    new_extended_lines.append([line2[0][0], line2[0][1], xi, yi])

                            else:
                                if s >= -1*extension_ratio and (-s)*line2[2]< maxgap:
                                    new_extended_lines.append([line1[0][2], line1[0][3], xi, yi])
                                    new_extended_lines.append([line2[0][2], line2[0][3], xi, yi])    

                else:
                    print("Invalid lines")
            # else:
            #     print("Invalid angle: ", abs_diff_in_angle)

    return new_extended_lines


def detect_ridges(gray, sigma=1.0):
    hxx, hyy, hxy = hessian_matrix(gray, sigma)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    return i1, i2

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

import cv2
from sort import *
mot_tracker = Sort() 


#cap = cv.VideoCapture('onbridge.MOV')
#ret, img1 = cap.read()
img = cv.imread('many_way.png') # trainImage

# scaledownh=1.0
# imgsample = cv.resize(img, (0,0), fx=(1/scaledownh), fy=(1/scaledownh)) 
# cv.imwrite("resize.bmp", imgsample)
imgsample = img.copy()
gray = cv.cvtColor(imgsample,cv.COLOR_BGR2GRAY)
cv.imwrite("filter_out_car_gray.bmp", gray)
# blur = cv.GaussianBlur(gray,(3,3),0)
# cv.imwrite("resize_gray_blur.bmp", blur)
#blur = cv.medianBlur(gray,3)

#edges = cv.Canny(gray,150,300,apertureSize = 3)
edges = cv.Canny(gray,200,300,apertureSize = 3)

frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pilimg = Image.fromarray(frame)
detections = detect_image(pilimg)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
img = np.array(pilimg)
pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
unpad_h = img_size - pad_y
unpad_w = img_size - pad_x
margin=4

mask = np.ones(edges.shape, dtype=np.uint8)*255
if detections is not None:
    tracked_objects = mot_tracker.update(detections.cpu())

    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
        cv2.rectangle(frame, (x1-margin, y1-margin), (x1+box_w+margin, y1+box_h+margin), (128, 128, 128), -1)
        cv2.rectangle(mask, (x1-margin, y1-margin), (x1+box_w+margin, y1+box_h+margin), 0, -1)

cv.imwrite("filter_out_car.bmp", frame)
cv.imwrite("filter_out_car_mask.bmp", mask)

# after_rot = imutils.rotate_bound(img2, 10)
# img2 = after_rot.copy()



maskout = cv.bitwise_and(edges, edges, mask=mask)   
edges = maskout.copy()
cv.imwrite("filter_out_car_gray_blur_canny.bmp", edges)
#cv.imshow("canny", edges)
#ridges = cv.ximgproc_RidgeDetectionFilter.getRidgeFilteredImage(gray)
# ridge_filter = cv.ximgproc.RidgeDetectionFilter_create(ksize=5)
# ridges = ridge_filter.getRidgeFilteredImage(gray)
#cv.imwrite("filter_out_car_gray_ridge.bmp", a)

# base_sigma=0.2
# max_sigma=20
# ridge_thd=128
# a, b = detect_ridges(gray, sigma=base_sigma)
# cv.normalize(b,  b, 0, 255, cv.NORM_MINMAX)
# cv.normalize(a,  a, 0, 255, cv.NORM_MINMAX)

# for k in range(2, max_sigma+1):
#     new_a, new_b = detect_ridges(gray, sigma=base_sigma*k)
#     cv.normalize(new_a,  new_a, 0, 255, cv.NORM_MINMAX)
#     cv.normalize(new_b,  new_b, 0, 255, cv.NORM_MINMAX)
#     b = np.minimum(b, new_b)
#     a = np.maximum(a, new_a)

# # cv.normalize(a,  a, 0, 255, cv.NORM_MINMAX)
# # cv.imwrite("filter_out_car_gray_ridges_white_ridge.bmp", a)
# cv.imwrite("filter_out_car_gray_dark_ridges_min.bmp",b)
# cv.imwrite("filter_out_car_gray_bright_ridges_max.bmp",a)

# maxval=255
# ret,thresh1 = cv.threshold(a,ridge_thd,maxval,cv.THRESH_BINARY)
# cv.imwrite("filter_out_car_gray_bright_ridges_max_threshold.bmp",thresh1)
# ret,thresh2 = cv.threshold(b,ridge_thd,maxval,cv.THRESH_BINARY_INV)
# cv.imwrite("filter_out_car_gray_dark_ridges_min_threshold.bmp",thresh2)



im1, contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#imgcon = imgsample.copy()
imgcon = np.zeros(edges.shape, dtype=np.uint8)
imgsamplecontour = imgsample.copy()
cv.drawContours(imgcon, contours, -1, 255, 7)
cv.drawContours(imgsamplecontour, contours, -1, (0,255,0), 7)
imgsamplecontourcopy = imgsamplecontour.copy()
cv.imwrite("filter_out_car_gray_blur_canny_contour.bmp", imgsamplecontour)

# im1_new, contours_new, hierarchy_new = cv.findContours(imgcon,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# imgsamplecontour_new = imgsample.copy()
# cv.drawContours(imgsamplecontour_new, contours_new, -1, (0,255,0), 7)
# cv.imwrite("filter_out_car_gray_blur_canny_contournew.bmp", imgsamplecontour_new)


#cv.imshow("contour", imgcon)


#imgsamplecontourbb = imgsample.copy()
aspect_ratio_thd=1.3
area_ratio=0.04
area_max=900
lines2=[]
allbb = imgsamplecontourcopy.copy()
chosenbb = imgsamplecontourcopy.copy()
for cnt in contours:
    area = cv.contourArea(cnt)
    rect = cv.minAreaRect(cnt)
    bbarea = rect[1][0]*rect[1][1]
    #rect = [(centerx, centery), width, height, angle]
    box = cv.boxPoints(rect)
    box = np.int0(box)

    if rect[1][0] > 0 and rect[1][1] > 0 :
        ratio = rect[1][1]/rect[1][0]
        if ratio < 1:
            ratio = 1/ratio

        bbarearatio = 1.0*area/bbarea
        if area < area_max and bbarearatio > area_ratio and ratio > aspect_ratio_thd:
            line = [(box[0][0]+box[3][0])*0.5, (box[0][1]+box[3][1])*0.5, (box[1][0]+box[2][0])*0.5, (box[1][1]+box[2][1])*0.5]
            line2 = [(box[0][0]+box[1][0])*0.5, (box[0][1]+box[1][1])*0.5, (box[3][0]+box[2][0])*0.5, (box[3][1]+box[2][1])*0.5]
            if linelength(line) > linelength(line2):
                lines2.append([line])
            else:
                lines2.append([line2])
            cv.drawContours(chosenbb,[box],0,(0,0,255),2)

    cv.drawContours(allbb,[box],0,(0,0,255),2)

cv.imwrite("filter_out_car_gray_blur_canny_contour_allbb.bmp", allbb)
cv.imwrite("filter_out_car_gray_blur_canny_contour_chosenbb.bmp", chosenbb)

#filter our perimeter/area ratio contour
# ratio_thd=1.5
# min_area_of_segment=15
# filtered_contours=[]
# for cnt in contours:
#         perimeter = cv.arcLength(cnt,True)
#         area = cv.contourArea(cnt)
#         if area > min_area_of_segment:
#                 ratio = perimeter/area
#                 if ratio > ratio_thd:
#                         filtered_contours.append(cnt)

# imgsamplecontourf = imgsample.copy()
# cv.drawContours(imgsamplecontourf, filtered_contours, -1, (0,255,0), 1)
# cv.imwrite("resize_gray_blur_canny_contour_filtered.bmp", imgsamplecontourf)  



# th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,13,0)
# cv.imshow("adaptive_thd", th3)

# minLineLength = 5
# maxLineGap = 5
# lines = cv.HoughLinesP(edges,1,np.pi/180,20,minLineLength,maxLineGap)
# imgsamplehlp = imgsample.copy()
# for line in lines:
#     cv.line(imgsamplehlp,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),6)
# #cv.imshow("houghlineP", imgsamplehlp)
# cv.imwrite("filter_out_car_gray_blur_canny_houghlinesP.bmp", imgsamplehlp)

# minLineLength = 20
# maxLineGap = 5
# lines2 = cv.HoughLinesP(imgcon,1,np.pi/180,100,minLineLength,maxLineGap)
# imgsamplehlp2 = imgsample.copy()

# for line in lines2:
#     cv.line(imgsamplehlp2,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),1)
# #cv.imshow("houghlineP", imgsamplehlp)
# cv.imwrite("filter_out_car_gray_blur_canny_contour_houghlinesP.bmp", imgsamplehlp2)

imgsamplebbline = imgsample.copy()
for line in lines2:
    cv.line(imgsamplebbline,(int(line[0][0]), int(line[0][1])),(int(line[0][2]), int(line[0][3])),(0,255,0),3)
#cv.imshow("houghlineP", imgsamplehlp)
cv.imwrite("filter_out_car_gray_blur_canny_contour_chosenbb_line.bmp", imgsamplebbline)

extended_lines = lines_merging(lines2)
imgsamplehlpex = imgsample.copy()
for line in extended_lines:
    cv.line(imgsamplehlpex,(int(line[0]),int(line[1])),(int(line[2]), int(line[3])),(0,255,0),3)
cv.imwrite("filter_out_car_gray_blur_canny_contour_chosenbb_mergecrack.bmp", imgsamplehlpex)

imgBA = np.zeros(edges.shape, dtype=np.uint8)
imgsamplehlpfinal = imgsample.copy()
for line in extended_lines:
    cv.line(imgsamplehlpfinal,(int(line[0]),int(line[1])),(int(line[2]), int(line[3])),(0,255,0),7)
    cv.line(imgBA,(int(line[0]),int(line[1])),(int(line[2]), int(line[3])), 255,7)

for cnt in contours:
    rect = cv.minAreaRect(cnt)
    area = rect[1][0]*rect[1][1]
    if area > 1000 and max(rect[1][0], rect[1][1])>300:
       cv.drawContours(imgsamplehlpfinal, [cnt], 0, (0,255,0), 7) 
       cv.drawContours(imgBA, [cnt], 0, 255, 7) 
cv.imwrite("filter_out_car_gray_blur_canny_contour_chosenbb_mergecrack_final.bmp", imgsamplehlpfinal)


colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(0,255,255),(255,255,255),(0,0,0),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
im1_new, contours_new, hierarchy_new = cv.findContours(imgBA,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
result = imgsample.copy()

obj_id=0                                                                                                                                                                    
for cnt in contours_new:
    rect = cv.minAreaRect(cnt)
    area = rect[1][0]*rect[1][1]
    if area > 1200 and max(rect[1][0], rect[1][1])>350:
        color = colors[int(obj_id) % len(colors)]
        obj_id=obj_id+1
        cv.drawContours(result, [cnt], 0, color, -1) 

cv.imwrite("filter_out_car_gray_blur_canny_contour_chosenbb_mergecrack_final_result.jpg", result)


obj_id=0
extend_margin_y=200                                                                                                                                                                         
for cnt in contours_new:
    rect = cv.minAreaRect(cnt)
    area = rect[1][0]*rect[1][1]
    if area > 1200 and max(rect[1][0], rect[1][1])>350:
        color = colors[int(obj_id) % len(colors)]
        obj_id=obj_id+1
        #cv.drawContours(result, [cnt], 0, color, -1) 

        #print("cnt: ", cnt)

        # Try polyfit to extend the lane segmentation result
        # get x and y vectors
        # domain knowledge, same y must have same x, but same x can have multiple y, so reverse the role of x and y
        x = cnt.squeeze()[:,0]
        y = cnt.squeeze()[:,1]                                                                                                                                       

        # calculate polynomial
        z = np.polyfit(y, x, 3)
        f = np.poly1d(z)

        # calculate new x's and y's
        y_new = np.linspace(min(y)-extend_margin_y, max(y)+extend_margin_y, 500)
        x_new = f(y_new)

        #new_cnt=np.hstack((x_new,y_new))
        new_cnt=[]
        for qk in range(0, len(x_new)-1):
            #new_cnt.append([[x_new[qk], y_new[qk]]])
            cv.line(result, (int(x_new[qk]), int(y_new[qk])), (int(x_new[qk+1]), int(y_new[qk+1])), color, 7) 

        #print("new_cnt: ", np.array(new_cnt))
        #cv.drawContours(result, [np.array(new_cnt).astype(np.int32)], 0, color, 7) 

cv.imwrite("filter_out_car_gray_blur_canny_contour_chosenbb_mergecrack_final_result_polyfit.jpg", result)



# HoughThd=70

# lines = cv.HoughLines(edges,1,4.0*np.pi/180,HoughThd)
# imgsamplehl = imgsample.copy()
# for i in  range(0,lines.shape[0]):
#     rho,theta = lines[i,0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv.line(imgsamplehl,(x1,y1),(x2,y2),(0,0,255),1)
# #cv.imshow("hough", imgsamplehl)
# cv.imwrite("filter_out_car_gray_blur_canny_houghlinesZ.bmp", imgsamplehl)

#Draw a mask
# mask = np.zeros(edges.shape, dtype=np.uint8)
# for i in  range(0,lines.shape[0]):
#     rho,theta = lines[i,0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv.line(mask,(x1,y1),(x2,y2),255,7)
#cv.imshow("mask", mask)


# maskout = cv.bitwise_and(edges, edges, mask=mask)     
#cv.imshow("maskout", maskout)
#cv.imwrite("maskout.bmp", maskout)


# cv.imshow("hough2", imgsample2)
# resized_frameh3 = cv.resize(imgsample, (0,0), fx=(scaledownh*0.3), fy=(scaledownh*0.3)) 
# resized_frameh4 = cv.resize(imgsample2, (0,0), fx=(scaledownh*0.3), fy=(scaledownh*0.3)) 
# combh = np.vstack((resized_frameh3, resized_frameh4))
# cv.imwrite("zzz_houghresult.bmp", combh)
# cv.imshow("houghresult", combh)


# imgsample2 = img1.copy()
# imgsample2 = cv.resize(imgsample2, (0,0), fx=(0.5), fy=(0.5)) 
# # Parameters
# # ddepth	Specifies output image depth. Defualt is CV_32FC1
# # dx	Order of derivative x, default is 1
# # dy	Order of derivative y, default is 1
# # ksize	Sobel kernel size , default is 3
# # out_dtype	Converted format for output, default is CV_8UC1
# # scale	Optional scale value for derivative values, default is 1
# # delta	Optional bias added to output, default is 0
# # borderType	Pixel extrapolation method, default is BORDER_DEFAULT

# ridge_filter = cv.ximgproc.RidgeDetectionFilter_create(ksize=3)
# ridges = ridge_filter.getRidgeFilteredImage(imgsample2)
# cv.imshow("ridges", ridges)

# imgsample3 = img1.copy()
# gray3 = cv.cvtColor(imgsample3,cv.COLOR_BGR2GRAY)
# gray3 = cv.resize(gray3, (0,0), fx=(0.5), fy=(0.5)) 



# a, b = detect_ridges(gray3, sigma=1.0)
# #cv.normalize(a,  a, 0, 255, cv.NORM_MINMAX)
# cv.imshow("ridges_max", a)
# #cv.normalize(b,  b, 0, 255, cv.NORM_MINMAX)
# cv.imshow("ridges_min", b)
#img2 = cv.imread('snapshot200.png') # trainImage
cv.waitKey(0)   
