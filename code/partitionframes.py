import scipy.io as sio
import numpy as np
import random

# This file splits up 'video.mat' into training & label sets
# We output 'training.mat' with the following variables
#	'train' -> the training dataset, shaped (num_samples, window_size, width, height)
#	'actions' -> a char array with the actions
#	'labels' -> the labels, shaped (num_samples, width, height)
# in this case, len(actions) = len(actions)

input_matr = sio.loadmat('video.mat')
video = input_matr['alist']
actions = input_matr['actions']
side_length = video.shape[2]
iter_length = video.shape[0]

num_train = 3
samples = iter_length - num_train

training_results = np.zeros((samples, num_train, side_length, side_length), dtype = np.float64)
test_results = np.zeros((samples, side_length, side_length), dtype = np.float64)
action_results = []

for i in range(0, iter_length - num_train):
	action_results.append(actions[i + num_train - 1])
	training_results[i, :, :, :] = video[i:i+num_train, :, :]
	test_results[i, :, :] = video[i + num_train, :, :]

sio.savemat("training.mat", {'train' : training_results, 'actions' : action_results, 'labels' : test_results })