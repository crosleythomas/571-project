from keras.models import Sequential
from keras.models import load_model

import time, sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
#plt.switch_backend('Qt4Agg')
matplotlib.use('qt4agg')
import matplotlib.image as mpimg

from PIL import Image

SCALE_FACTOR = 255.0

data_file = 'sprites/sprites_empty_baseline.npz'
data_file = sys.argv[1] if len(sys.argv) > 1 else data_file

data = np.load(data_file)
inputs = data['frames']
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
input_handle = fig.add_subplot(1,1,1)
input_handle.set_title('Input Image')

for i in range(0, display_iters - 1):
	# Input
	# If using multiple frames the last (-1) frame should be the previous frame
	frame_input = inputs[i,-1,:,:]
	frame_input = np.reshape(frame_input, [frame_height, frame_width])
	action_input = actions[i]
	action_input = np.reshape(actions[i], [1, action_dim])

	# Display
	input_handle = fig.add_subplot(1,1,1)

	if i == 0:
		ihandle = input_handle.imshow(frame_input)
	else:
		ihandle.set_data(frame_input)

	fig.show()
	ans = raw_input("Press [enter] to continue, type anything to exit.")
	if len(ans) > 0:
		sys.exit(0)
