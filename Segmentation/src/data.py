import os
import cv2
import gc
import numpy as np 
import pandas as pd 
import nibabel as nib

# data preprocessing

PATH = 'data/metadata.csv'

meta = pd.read_csv(PATH)

# Crop images
def cropper(test_img):
    
    '''
    Crop images on the basis of their contours to avoid lose of information
    and remove noise.

    Arguments:
    test_img -- input image to be resized

    Return:
    points_lung1 -- 
    points_lung2 -- 
    '''

    test_img = test_img*255
    test_img = np.uint8(test_img)

    contours,hierarchy = cv2.findContours(test_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    x = np.argsort(areas)

    max_index = x[x.size - 1]
    cnt1=contours[max_index]
    second_max_index = x[x.size - 2]
    cnt2 = contours[second_max_index]

    x,y,w,h = cv2.boundingRect(cnt1)
    p,q,r,s = cv2.boundingRect(cnt2)

    cropped1 = test_img[y:y+h, x:x+w]
    cropped1 = cv2.resize(cropped1, dsize=(112,224), interpolation=cv2.INTER_AREA)
    cropped2 = test_img[q:q+s, p:p+r]
    cropped2 = cv2.resize(cropped2, dsize=(112,224), interpolation=cv2.INTER_AREA)

    points_lung1 = []
    points_lung2 = []

    points_lung1.append(x); points_lung1.append(y); points_lung1.append(w); points_lung1.append(h)
    points_lung2.append(p); points_lung2.append(q); points_lung2.append(r); points_lung2.append(s)

    return(points_lung1, points_lung2)

def prepare(num=1):

    '''
    Prepare Image data.
    ------------------------------------------
    Remove noise and reshape them (B, W, H, C).
    Crop images on the basis of lung contours. 
    Apply CLAHE enhancer and resize images.
    
    Arguments:
    num -- number of observation in .csv file.
     
    Returns:
    cts -- list of ct scans.
    infection -- list of corresponding infection mask.
    '''
    
    cts = []
    infections = []

    for i in range(num=1):

        lung = np.rot90(nib.load(meta.loc[i, 'lung_mask']).get_fdata())
        ct = np.rot90(nib.load(meta.loc[i, 'ct_scan']).get_fdata())
        infection = np.rot90(nib.load(meta.loc[i, 'infection_mask']).get_fdata())
        
        slices = lung.shape[2]
        
        lung = lung[:,:,round(slices*0.2):round(slices*0.8)]
        ct = ct[:,:,round(slices*0.2):round(slices*0.8)]
        infection = infection[:,:,round(slices*0.2):round(slices*0.8)]
        
        lung_img = np.reshape(np.rollaxis(lung, 2), (lung.shape[2], lung.shape[0], lung.shape[1]))
        ct_img = np.reshape(np.rollaxis(ct, 2), (ct.shape[2],ct.shape[0],ct.shape[1]))
        infection_img = np.reshape(np.rollaxis(infection, 2), (infection.shape[2],infection.shape[0],infection.shape[1]))

        for num in range(0, lung_img.shape[0]):
            
            lung = lung_img[num]
            ct = ct_img[num]
            infection = infection_img[num]
            
            
            points1, points2 = cropper(lung)
        
            a,b,c,d = points1
            e,f,g,h = points2

            img1 = ct[b:b+d, a:a+c]
            img1 = cv2.resize(img1, dsize=(112, 224), interpolation=cv2.INTER_AREA)
            img2 = ct[f:f+h, e:e+g]
            img2 = cv2.resize(img2, dsize=(112, 224), interpolation=cv2.INTER_AREA)
            ct = np.concatenate((img1, img2), axis=1)
            
            infec = np.uint8(infection*255)
            img3 = infec[b:b+d, a:a+c]
            img3 = cv2.resize(img3, dsize=(112,224), interpolation=cv2.INTER_AREA)
            img4 = infec[f:f+h, e:e+g]
            img4 = cv2.resize(img4, dsize=(112,224), interpolation=cv2.INTER_AREA)
            infection = np.concatenate((img3, img4), axis=1)
            
        
            cts.append(ct)
            infections.append(infection)
            
            gc.collect()

    return cts, infections