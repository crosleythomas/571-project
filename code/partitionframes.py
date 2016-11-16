import scipy.io as sio
import numpy as np
import random

input_matr = sio.loadmat('video.mat')
video = input_matr['alist']
actions = input_matr['actions']
side_length = video.shape[0]
iter_length = video.shape[2]

num_train = 3
samples = iter_length - num_train

train_frames = np.zeros((side_length, side_length, num_train), dtype = np.float64)
test_frame = np.zeros((side_length, side_length), dtype = np.float64)

training_results = np.zeros((side_length, side_length, num_train, samples), dtype = np.float64)
test_results = np.zeros((side_length, side_length, samples), dtype = np.float64)
action_results = []

for i in range(0, iter_length - num_train):
	action_results.append(actions[i + num_train - 1])
	train_frames[:, :, :] = video[:, :, i:i+num_train]
	test_frame = video[:, :, i + num_train]
	training_results[:, :, :, i] = train_frames
	test_results[:, :, i] = test_frame

sio.savemat("training.mat", {'train' : training_results, 'actions' : action_results, 'labels' : test_results })