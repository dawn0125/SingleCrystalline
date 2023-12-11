# -*- coding: utf-8 -*-
"""
procedure: 
    1. detect red cracks and houghlines, then blur and ignore all those parts 
    2. sharpen image like crazy and detect yellow horizontal lines with houghlines 
"""

import numpy as np 
import cv2 as cv 
import os 
import matplotlib.pyplot as plt
from scipy import ndimage 
from rembg import remove 

#=============================FUNCTIONS========================================
    
def threshManual(img, lower, upper):
    '''
    thresh according to bins added manually 
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, lower, upper, cv.THRESH_BINARY_INV)
    return thresh
    

def threshOtsu(img):
    '''
    img: image array 
    thresh: black and white image array 

    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return thresh

def findContours(img):
    '''
    img: image array 
    contours: image array of just the contours of img 

    '''
    if np.ndim(img) != 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def findAreas(contours):
    '''
    contours: image array with contours found 
    area: array with area of each contour 

    '''
    area = []
    for cnt in contours:
        area.append(cv.contourArea(cnt))
    return np.asarray(area)

def redMask(img, thresh):       
    '''
    returns a mask 
    '''     
    cnts, hierarchies = findContours(thresh)
    areas = findAreas(cnts)            
            
    sample_mask = remove(img, post_process_mask = True, only_mask = True)
    mask = np.full((img.shape[0], img.shape[1]), fill_value=0, dtype="uint8")
    
    # draw mask of obvious flaws 
    for j in np.arange(len(areas)):
        cv.drawContours(mask, cnts, j, color = 1, thickness = cv.FILLED)
    
    # combine contour mask and sample mask 
    mask = cv.bitwise_and(mask, sample_mask)
    
    redx, redy = np.where(mask > 0)
    opaque = img.copy()
    opaque[redx, redy, :] = [0, 0, 255]
    
    return opaque 

#=============================MAIN========================================
# directories
img_directory = '//wp-oft-nas/HiWis/GM_Dawn_Zheng/Vurgun/SX/Cropped'
outdir = '//wp-oft-nas/HiWis/GM_Dawn_Zheng/Vurgun/SX/threshim'

#  parameters 
loi = os.listdir(img_directory)
acceptedFileTypes = ['tif'] # add more as needed 
opacity = 0.25
show_plot = True

# threshing 
manual_threshing = True
lower = 153
upper = 181



for i in loi:   
    if( '.' in i and i.split('.')[-1] in acceptedFileTypes):
        
        # open image 
        f = img_directory + '/' + i
        img = cv.imread(f)
        
        # process image 
        blur = ndimage.gaussian_filter(img, 10, mode='nearest')
        
        if manual_threshing == True: 
            thresh = threshManual(blur, lower, upper)
        elif manual_threshing == False:
            thresh = threshOtsu(blur)
        
        # get blemishes 
        opaque = redMask(img, thresh)
        # transluscent is just an output image 
        transluscent = cv.addWeighted(img, 1-opacity, opaque, opacity, 0)
        
        if show_plot == True:
            plt.subplot(131), plt.imshow(opaque)
            plt.title('opaque')
            plt.subplot(132), plt.imshow(transluscent)
            plt.title('transluscent')
        
        # find scratches 
        edges = cv.Canny(opaque, 153, 180)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength = 100, maxLineGap = 10)
        
        # draw scratches 
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(transluscent, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        if show_plot == True:
            plt.subplot(133),plt.imshow(transluscent)
            plt.title('scratches')
            plt.show()
            
        # cv.imwrite(outdir + '/' + i, scratches)