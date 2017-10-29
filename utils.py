import tensorflow as tf
import numpy as np
import optparse
import os
import shutil
import time
import random
import sys
import pickle
import glob
import matplotlib.pyplot as plt

from layers import *

from scipy.misc import imsave
from scipy.misc import imresize
from PIL import Image
from tqdm import tqdm
from numpy import genfromtxt



def normalize_data(data):

	return (data - np.mean(data, axis=1, keepdims=True))/np.std(data, axis=1, keepdims=True)

def divide_data(input_data):

	size = np.shape(input_data)

	num_train_days = (int)(0.9*size[1])

	
	train_data = input_data[:, :num_train_days]
	test_data = input_data[:, num_train_days:]

	train_data = np.reshape(train_data, [-1, 1])
	test_data = np.reshape(test_data, [-1, 1])

	return train_data, test_data

def pre_process(data, data_len, feature_size):

	input_mat = np.zeros([data_len - feature_size, feature_size, 1], np.float32)
	output_mat = np.zeros([data_len - feature_size, 1, 1], np.float32)
	mean_mat = np.zeros([data_len - feature_size, 1, 1], np.float32)
	std_mat = np.zeros([data_len - feature_size, 1, 1], np.float32)

	for i  in range(data_len - feature_size):
		input_mat[i] = data[i:i+feature_size]
		output_mat[i] = data[i+feature_size:i+feature_size+1]

	mean_mat = np.mean(input_mat, axis=1, keepdims=True)
	std_mat = np.std(input_mat, axis=1, keepdims=True)

	# print(std_mat)

	input_mat = (input_mat - mean_mat)/(std_mat)
	output_mat = (output_mat - mean_mat)/(std_mat)

	return input_mat, output_mat
