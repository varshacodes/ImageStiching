#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 5 16:47:55 2019

@author: varshaganesh
"""
import cv2
import os
from scipy.spatial.distance import cdist
import numpy as np


def getWarp(img, Homograph):
    
    """ 
        Computes the warpPerspective of a Image using Open CV's warp Perspective method and returns the same
    """

    out_size = (1200,800)

    warpedImg = cv2.warpPerspective(img, Homograph, out_size)

    warpedImg = np.uint8(warpedImg)

    return warpedImg

def ransacHomography(matches, locs1, locs2, iterations=10000, tolerance=2):

    """
        matches refer to the matches between the two sets of point locations loc1 and loc2
    
        This method Computes the best Homography between the Two Images using RANSAC Technique
        
        I've used 10000 iterations to compute the Homography and also a tolerance of 2

    """
    
    Homo_12 = np.zeros((3,3))

    maxInliers = -1

    bestInliers = np.zeros((1,1))

    p1 = locs1[matches[:,0], 0:2].T

    p2 = locs2[matches[:,1], 0:2].T



    for i in range(0, 1000):

        idx = np.random.choice(len(matches), 4)

        rand1 = p1[:,idx]
        rand2 = p2[:,idx]

        H = computeH(rand1, rand2)



        p2_est = np.append(p2.T, np.ones([len(p2.T),1]),1)

        p2_est = p2_est.T

        p1_est = np.matmul(H,p2_est)

        p1_est = p1_est/p1_est[2,:]



        actual_diff = np.square(p1[0,:] - p1_est[0,:]) + np.square(p1[1,:] - p1_est[1,:])

        inliers = actual_diff < tolerance**2

        numInliers = sum(inliers)

        

        if numInliers > maxInliers:

            maxInliers = numInliers

            bestInliers = inliers



    Homo_12 = computeH(p1[:,bestInliers], p2[:,bestInliers])

    return Homo_12


def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    """
        Creates a Gaussian Pyramid inorder to find the difference between two Images
    """
    if len(im.shape)==3:

        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if im.max()>10:

        im = np.float32(im)/255

    gaussian = []

    for i in levels:

        sigma_ = sigma0*k**i 

        gaussian.append(cv2.GaussianBlur(im, (0,0), sigma_))

    gaussian = np.stack(gaussian, axis=-1)

    return gaussian


def createDoGPyramid(gaussian, levels=[-1,0,1,2,3,4]):

    """
        Creates A Dog Pyramid using the difference in Gaussian Pyramid Inputs
    """

    dog = []

    dog_levels = levels[1:]

    for i in range(1, len(dog_levels)+1):

        dog.append(gaussian[:,:,i] - gaussian[:,:,i-1])



    dog = np.stack(dog, axis=-1)

    return dog, dog_levels



def computePrincipalCurvature(dog):

    """

        Takes in the generated Dog pyramid ad input and returns

        PrincipalCurvature,i.e, a matrix  where each point contains 
        
        curvature ratio R for the corre-sponding point in the Dog pyramid
        
    """


    pCurvature = np.zeros((dog.shape[0], dog.shape[1], dog.shape[2]))


    for i in range(0, dog.shape[2]):

        sobelx = cv2.Sobel(dog[:,:,i],cv2.CV_64F,1,0,ksize=3)

        sobely = cv2.Sobel(dog[:,:,i],cv2.CV_64F,0,1,ksize=3)

        sobelxx = cv2.Sobel(sobelx,cv2.CV_64F,1,0,ksize=3)

        sobelyy = cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=3)

        sobelxy = cv2.Sobel(sobelx,cv2.CV_64F,0,1,ksize=3)

        sobelyx = cv2.Sobel(sobely,cv2.CV_64F,1,0,ksize=3)

        traceH = np.square(np.add(sobelxx,sobelyy))

        detH = np.subtract(np.multiply(sobelxx, sobelyy), np.multiply(sobelxy,sobelyx))

        pCurvature[:,:,i] = np.divide(traceH, detH)

        pCurvature[:,:,i] = np.nan_to_num(pCurvature[:,:,i])

    return pCurvature



def getLocalExtrema(dog, dog_levels, pCurvature,th_contrast=0.03, th_r=12):

    """
          Returns local extrema points in both scale and space using the DoGPyramid

    """

    imh, imw, iml = dog.shape

    extrema = np.zeros((11,imh,imw,iml))

    for layer in range(0, iml):

        temp = np.pad(dog[:,:,layer],(1,1),mode='constant',constant_values=0)

        extrema[0,:,:,layer] = np.roll(temp,1,axis=1)[1:-1,1:-1] #right

        extrema[1,:,:,layer] = np.roll(temp,-1,axis=1)[1:-1,1:-1] #left

        extrema[2,:,:,layer] = np.roll(temp,1,axis=0)[1:-1,1:-1] #down

        extrema[3,:,:,layer] = np.roll(temp,-1,axis=0)[1:-1,1:-1] #up

        extrema[4,:,:,layer] = np.roll(np.roll(temp, 1, axis=1),1,axis=0)[1:-1,1:-1] #right,down

        extrema[5,:,:,layer] = np.roll(np.roll(temp, -1, axis=1),1,axis=0)[1:-1,1:-1] #left,down

        extrema[6,:,:,layer] = np.roll(np.roll(temp, -1, axis=1),-1,axis=0)[1:-1,1:-1] #left,up

        extrema[7,:,:,layer] = np.roll(np.roll(temp, 1, axis=1),-1,axis=0)[1:-1,1:-1] #right,up

        if layer == 0:

            extrema[9,:,:,layer] = dog[:,:,layer+1] #layer above

        elif layer == iml-1:

            extrema[8,:,:,layer] = dog[:,:,layer-1] #layer below

        else:

            extrema[8,:,:,layer] = dog[:,:,layer-1] #layer below

            extrema[9,:,:,layer] = dog[:,:,layer+1] #layer above

        extrema[10,:,:,layer] = dog[:,:,layer]



    extremas = np.argmax(extrema, axis=0)

    extremaPoints = np.argwhere(extremas==10)

    locsDoG = []



    for point in extremaPoints:

        if np.absolute(dog[point[0],point[1],point[2]]) > th_contrast and pCurvature[point[0],point[1],point[2]] < th_r:

            point = [point[1], point[0], point[2]]

            locsDoG.append(point)



    locsDoG = np.stack(locsDoG, axis=-1)

    locsDoG = locsDoG.T

    return locsDoG



def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], th_contrast=0.03, th_r=12):

    gaussian = createGaussianPyramid(im, sigma0, k, levels)

    dog, dog_levels = createDoGPyramid(gaussian, levels)

    p_curvature = computePrincipalCurvature(dog)

    locsDoG = getLocalExtrema(dog, dog_levels, p_curvature,th_contrast,th_r)

    return locsDoG, gaussian




def makeTestPattern(patch_width=9, nbits=256):

    """ 
        Creates Test for Brief
    """

    lin_combinations = patch_width**2

    compareX = np.random.choice(lin_combinations, nbits).reshape((nbits,1))

    compareY = np.random.choice(lin_combinations, nbits).reshape((nbits,1))

    test_pattern_file = '../temp/tempdata.npy'
    if not os.path.isdir('../temp'):

        os.mkdir('../temp')

    np.save(test_pattern_file, [compareX, compareY])

    return  compareX, compareY


test_pattern_file = '../temp/tempdata.npy'

if os.path.isfile(test_pattern_file):

    compareX, compareY = np.load(test_pattern_file)

else:

    compareX, compareY = makeTestPattern(9,256)

    if not os.path.isdir('../temp'):

        os.mkdir('../temp')
    np.save(test_pattern_file, [compareX, compareY])
    
    
def computeBrief(im, gaussian_pyramid, locsDoG, k, levels, compareX, compareY):

    """
            Computes Brief feature
    """
    desc = []

    locs = []

    for point in locsDoG:

        layer = point[2]

        im = gaussian_pyramid[:,:,layer]

        x = point[1]

        y = point[0]

        impatch = im[x-4:x+5,y-4:y+5]

        P = impatch.transpose().reshape(-1)

        if P.shape[0] < 81:

            continue

        else:

            im_desc = []

            for (x,y) in zip(compareX, compareY):

                if P[x] < P[y]:

                    im_desc.append(1)

                else:

                    im_desc.append(0)

            if len(im_desc) > 0:

                desc.append(im_desc)

                locs.append(point)



    locs = np.stack(locs, axis=-1)

    desc = np.stack(desc, axis=-1)

    locs = locs.T

    desc = desc.T

    return locs, desc



def getBrief(img):

    locsDoG, gauss_pyramid = DoGdetector(img)

    DoG_pyramid, levels = createDoGPyramid(gauss_pyramid)

    test_pattern_file = '../temp/tempdata.npy'

    if os.path.isfile(test_pattern_file):

        compareX, compareY = np.load(test_pattern_file)

    locs, desc = computeBrief(img, DoG_pyramid, locsDoG, np.sqrt(2), levels, compareX, compareY)

    return locs, desc



def getbriefMatch(desc1, desc2, ratio=0.8):

    """
        performs brief matching and returns the same
    """

    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')

    ix2 = np.argmin(D, axis=1)

    d1 = D.min(1)

    d12 = np.partition(D, 2, axis=1)[:,0:2]

    d2 = d12.max(1)

    r = d1/(d2+1e-10)

    is_discr = r<ratio

    ix2 = ix2[is_discr]

    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)

    return matches


def computeH(p1, p2):

    assert(p1.shape[1]==p2.shape[1])

    assert(p1.shape[0]==2)

    A = np.zeros((2*p1.shape[1],9))

    p1 = p1.T

    p2 = p2.T

    length = p1.shape[0]

    for i in range(0,length):

        u,v = p1[i,0], p1[i,1]

        x,y = p2[i,0], p2[i,1]

        A[i*2,:] = np.array([-x,-y,-1,0,0,0,x*u,y*u,u])

        A[i*2+1,:] = np.array([0,0,0,-x,-y,-1,v*x,v*y,v])



    [D,V] = np.linalg.eig(np.matmul(A.T,A))

    idx = np.argmin(D)

    Homo_12 = np.reshape(V[:,idx], (3,3))

    return Homo_12








