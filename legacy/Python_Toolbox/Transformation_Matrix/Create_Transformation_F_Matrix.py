# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:43:44 2018

@author: Umer Sherdil Paracha
"""
import numpy as np

def create_transformation_f_matrix(R, W):
    
    inverse_term = np.linalg.inv(R.transpose().dot(W).dot(R))
    T_f = W.dot(R).dot(inverse_term).transpose()
    
    return T_f