
from keras.models import Sequential
from keras.models import load_model

import time, sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg')
import matplotlib.image as mpimg

from PIL import Image

SCALE_FACTOR = 255.0

data_file = 'sprites/sprites_training_dots.npz'
data_file = sys.argv[1] if len(sys.argv) > 1 else data_file

data = np.load(data_file)
inputs = data['frames']
labels = data['labels']
actions = data['actions']

num_frames = inputs.shape[0]
input_channels = inputs.shape[1]
frame_height = inputs.shape[2]
frame_width = inputs.shape[3]

action_dim = 1

# Display side-by-side the predicted frame and true frame
display_iters = num_frames

# Set up image display
plt.ion()
fig = plt.figure()
input_handle = fig.add_subplot(1,2,1)
input_handle.set_title('Input Image')
next_handle = fig.add_subplot(1,2,2)
next_handle.set_title('Next Image')

for i in range(0, display_iters):
	# Input
	# If using multiple frames the last (-1) frame should be the previous frame
	frame_input = inputs[i,-1,:,:]
	frame_input = np.reshape(frame_input, [frame_height, frame_width])
	action_input = actions[i]
	action_input = np.reshape(actions[i], [1, action_dim])

	# True image
	next_frame = labels[i,:,:,:]
	next_frame = np.reshape(next_frame, [frame_height, frame_width])
	next_frame = next_frame * SCALE_FACTOR

	# Display
	input_handle = fig.add_subplot(1,2,1)
	next_handle = fig.add_subplot(1,2,2)

	if i == 0:
		ihandle = input_handle.imshow(frame_input)
		nhandle = next_handle.imshow(next_frame)
	else:
		ihandle.set_data(frame_input)
		nhandle.set_data(next_frame)

	fig.show()
	ans = raw_input("Press [enter] to continue, type anything to exit.")
	if len(ans) > 0:
		sys.exit(0)
