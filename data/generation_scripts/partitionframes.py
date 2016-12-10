import scipy.io as sio
import numpy as np
import random

# This file splits up baseline data file into input & label sets
# We output 'training.mat' with the following variables
#	'input' -> the training dataset, shaped (num_samples, frames_per_input * num_channels, width, height)
#	'actions' -> a char array with the actions
#	'labels' -> the labels, shaped (num_samples, width, height)
# in this case, len(actions) = len(actions)

create_mat_file = 0
create_npz_file = 1

baseline_file = '../sprites/sprites_baseline_data_dots.npz'
output_file = '../sprites/single_input_sprites_dots_training'

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

for i in range(0, samples):
	actions.append(base_actions[i + frames_per_input - 1])
	sample = video[i:i+frames_per_input, :, :, :]
	frames[i, :, :, :] = np.reshape(sample, (1, total_channels, frame_height, frame_width))
	labels[i, :, :, :] = video[i + frames_per_input, :, :, :]

if create_mat_file:
	sio.savemat(output_file, {'train' : frames, 'actions' : actions, 'labels' : labels })

if create_npz_file:
	np.savez(output_file + '.npz', frames=frames, actions=actions, labels=labels)