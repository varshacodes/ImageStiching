#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 5 16:47:55 2019

@author: varshaganesh
"""
import cv2
import os
import argparse
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import utils 

def stichImages(img1, img2, Homo_12):

    """
        Stiches Images img1 and img2 without cliping
    
        using the computed homography matrix of these images .
    
    """
    img_h1, img_w1, img_d1 = img1.shape

    img_h2, img_w2, img_d2 = img2.shape

    corners = np.array([[0,img_w2,0,img_w2],[0,0,img_h1,img_h1],[1,1,1,1]])

    warpedCorners = np.matmul(Homo_12, corners)

    warpedCorners = warpedCorners/warpedCorners[2,:]

    warp_corner = np.ceil(warpedCorners)

    cols = img2.shape[1]

    minrow = min(1,min(warp_corner[1,:]))

    maxcol = max(cols,max(warp_corner[0,:]))

    mincol = min(1,min(warp_corner[0,:]))

    W_out = 2000

    height = 2000

    out_size = (W_out, height)

    s = W_out / (maxcol-mincol)

    scaleM = np.array([[s,0,0],[0,s,0],[0,0,1]])

    transM = np.array([[1,0,0],[0,1,-minrow],[0,0,1]])

    M = np.matmul(scaleM,transM)


    img2Warped = cv2.warpPerspective(img2, np.matmul(M,Homo_12), out_size)

    img1Warped = cv2.warpPerspective(img1, np.matmul(scaleM,transM), out_size)



    mask1 = distance_transform_edt(img1Warped)

    mask2 = distance_transform_edt(img2Warped)



    result1 = np.multiply(img1Warped,mask1)

    result2 = np.multiply(img2Warped,mask2)



    pano_im = np.divide(np.add(result1, result2), np.add(mask1, mask2))

    pano_im = np.nan_to_num(pano_im)

    pano_im = np.uint8(pano_im)

    return pano_im


def getHomography(img1,img2):
    
    """ 
        Computes the Homography between Img1 and Img2 and returns the same
    """

    locs1, desc1 = utils.getBrief(img1)

    locs2, desc2 = utils.getBrief(img2)

    matches = utils.getbriefMatch(desc1, desc2)

    Homograph = utils.ransacHomography(matches, locs1, locs2, iterations=10000, tolerance=2)
    
    return Homograph



def getPanaroma(ImgFolder,Img1,Img2,Img3):
    
    """ 
        Given: 3 Images img1, img2 and img3
    
        STEP 1: Compute Homography between img1 and img2, call it Homo_12
        
        STEP 2: Compute Homography between img2 and img3, call it Homo_23
        
        STEP 3: Stich img2 and img3 using  Homo_23, call it pano23
        
        STEP 4: Compute Homographic composition of Homo_12 and Homo_23, call it Homo_123
        
        STEP 5: Use Homo_123 to warp img1, call it warpedImg
        
        STEP 6: Stich the warpedImg with Pano23
        
        STEP 7: Write the result to the desired folder as panorama.jpg
    """
    
    img1 = cv2.imread(Img1)
    
    img2 = cv2.imread(Img2)
    
    img3 = cv2.imread(Img3)
    
    Homo_12 = getHomography(img1,img2)
    
    Homo_23 = getHomography(img2,img3)
    
    Homo_123 = np.matmul(Homo_12,Homo_23)
    
    warpedImg = utils.getWarp(img1, Homo_123)
    
    pano23 = stichImages(img2,img3,Homo_23)
    
    panorama = stichImages(warpedImg,pano23,Homo_123)
    
    cv2.imwrite(ImgFolder+'panorama.jpg',panorama)
    
    
def getPanaroma2(ImgFolder,Img1,Img2):
    
    """ Given: 2 Images img1, img2 
    
        STEP 1: Compute Homography between img1 and img2, call it Homo_12
                
        STEP 2: Stich img1 and img2 using  Homo_12, call it panorama
        
        STEP 3: Write the result to the desired folder as panorama.jpg
        
    """
    
    img1 = cv2.imread(Img1)
    
    img2 = cv2.imread(Img2)
    
    Homo_12 = getHomography(img1,img2)
    
    panorama = stichImages(img1,img2,Homo_12)
    
    cv2.imwrite(ImgFolder+'panorama.jpg',panorama)   
    
    
def parse_args():
    
    """ Function to Parse the Command Line Argument which specifies the Data Directory"""
    
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    
    parser.add_argument('string', type=str, default="./ubdata/",help="Resources folder,i.e, folder in which images are stored")
    
    args = parser.parse_args()
    
    return args   


def getImages(folder):
    
    """ Function to Retreive the Img files with Extension .jpg from the desired Folder"""
    
    images = []
    
    for filename in os.listdir(folder):
        
        if filename.endswith(".jpg") and not filename ==("panorama.jpg"):
            
            images.append(folder + filename)
            
    images.sort()
    
    return images


if __name__ == '__main__':
    
    args = parse_args()
    
    ImgsFolder = args.string
    
    Imgs = getImages(ImgsFolder)
    
    print(Imgs)
    
    """ 
        If the Number of Images to be stiched is 2, we perform a simple stich using getPanaroma2 Method
    """
    
    if len(Imgs)==3:
        
        getPanaroma(ImgsFolder,Imgs[0],Imgs[1],Imgs[2])
        
    """ 
        Else if the Number of Images to be stiched is 3, we perform the stich using getPanaroma Method 
    """
        
    if(len(Imgs)==2):
        
        getPanaroma2(ImgsFolder,Imgs[0],Imgs[1])
        
    print("success")
