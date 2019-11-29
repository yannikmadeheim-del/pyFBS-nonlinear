# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:43:44 2018

@author: Umer Sherdil Paracha
"""
import numpy as np

def create_transformation_u_matrix(R, W):
    
    return np.linalg.inv(R.transpose().dot(W).dot(R)).dot(R.transpose()).dot(W)