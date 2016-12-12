import scipy.io as sio
import numpy as np
import random
import sys
# This file splits up baseline data file into input & label sets
# We output 'training.mat' with the following variables
#	'input' -> the training dataset, shaped (num_samples, frames_per_input * num_channels, width, height)
#	'actions' -> a char array with the actions
#	'labels' -> the labels, shaped (num_samples, width, height)
# in this case, len(actions) = len(actions)

create_mat_file = 0
create_npz_file = 1
one_hot = True

baseline_file = '../sprites/sprites_walled_baseline.npz'
baseline_file = sys.argv[1] if len(sys.argv) > 1 else baseline_file
output_file = '../sprites/sprites_walled_baseline_training'
output_file = sys.argv[2] if len(sys.argv) > 2 else output_file

data = np.load(baseline_file)
video = data['frames']
base_actions = data['actions']

num_frames = video.shape[0]
num_channels = video.shape[1]
frame_height = video.shape[2]
frame_width = video.shape[3]

frames_per_input = 1
total_channels = frames_per_input * num_channels

samples = num_frames - frames_per_input

total_channels = frames_per_input * num_channels

frames = np.zeros((samples, total_channels, frame_height, frame_width), dtype = np.float64)
labels = np.zeros((samples, num_channels, frame_height, frame_width), dtype = np.float64)
actions = []


#####################################
### Setup arguments based on data ###
#####################################


for i in range(0, samples):
	actions.append(base_actions[i + frames_per_input - 1])
	sample = video[i:i+frames_per_input, :, :, :]
	frames[i, :, :, :] = np.reshape(sample, (1, total_channels, frame_height, frame_width))
	labels[i, :, :, :] = video[i + frames_per_input, :, :, :]

one_hot_vectors = None
one_hot_actions = None
if one_hot:
	one_hot_vectors = {}
	action_set = set(actions)
	action_size = len(action_set)
	vector_index = 0
	for action in action_set:
		one_hot_vector = [0] * action_size
		one_hot_vector[vector_index] = 1
		one_hot_vectors[action] = one_hot_vector
		vector_index += 1
	one_hot_actions = [ one_hot_vectors[action] for action in actions ]

if create_mat_file:
	sio.savemat(output_file, {'train' : frames, 'actions' : one_hot_actions if one_hot else actions, 'labels' : labels })

if create_npz_file:
	np.savez(output_file + '.npz', frames=frames, actions=actions, labels=labels)