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

#=============================FUNCTIONS========================================
    
def threshManual(img, lower, upper):
    '''
    thresh according to bins added manually 
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, lower, upper, cv.THRESH_BINARY)
    return thresh
    

def threshOtsu(img):
    '''
    img: image array 
    thresh: black and white image array 

    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
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
    
#=============================MAIN========================================
# directories
img_directory = '//wp-oft-nas/HiWis/GM_Dawn_Zheng/Vurgun/SX/Neuer Ordner'
outdir = '//wp-oft-nas/HiWis/GM_Dawn_Zheng/Vurgun/SX/threshim'
#  parameters 
loi = os.listdir(img_directory)
acceptedFileTypes = ['tif'] # add more as needed 

# threshing 
manual_threshing = True
lower = 125
upper = 175


for i in loi:   
    if( '.' in i and i.split('.')[-1] in acceptedFileTypes):
        f = img_directory + '/' + i
        img = cv.imread(f)
        testim = img.copy()
        
        plt.imshow(img)
        plt.show()
        
        bilateral = cv.bilateralFilter(img, 5, 75, 75)
        if manual_threshing == True: 
            thresh = threshManual(img, lower, upper)
        elif manual_threshing == False:
            thresh = threshOtsu(bilateral)
        
        plt.imshow(thresh)
        plt.show()
        
        # cnts, hierarchies = findContours(thresh)
        # areas = findAreas(cnts)
        # a = np.argsort(areas)[-1]
        # parent = hierarchies[:, :, 3][0]
        
        # for j in np.arange(len(areas)):
        #     if areas[j] >= 500: 
        #         cv.drawContours(testim, cnts[j], -1, (0, 255, 255), 10)
        #         plt.imshow(testim)
        #         plt.show()
        
        cv.imwrite(outdir + '/' + i, testim)
        
        