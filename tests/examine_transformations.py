#!/usr/bin/env python
####################################################################
# Implementation of Oh, et al. action-conditoinal video prediction #
#	in keras.													   #
#																   #
# Test to examine hwo the action transformation branch affects the #
# 	predictions.  Pass in all ones from the A.T. branch such that  #
#   no transformation happens in the point-wise product.		   #
#																   #
# UW CSE 571 Project											   #
# Thomas Crosley and Karan Singh								   #
####################################################################

# Keras Imports
from keras.models import *
from keras.layers import *
from keras.utils.visualize_util import plot
from keras.models import model_from_yaml

# Other Imports
import scipy.io as sio
import numpy as np
import time, datetime
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg')
import matplotlib.image as mpimg
from PIL import Image

SCALE_FACTOR = 255.0
np.set_printoptions(threshold=np.nan)

#######################################
### Load a model we want to examine ###
#######################################
model_file = '../data/trained_models/sprites_training_ac_ff_2016-12-08_20:07:06.h5'
model = load_model(model_file)

action_projection_layer_index = 7

###############################
###        Load data        ###
###############################

# Load frame data as numpy arrays
#	Input has to be arranged such that, even if the input for a single training
#	example has multiple concatenated frames, all frames are a grouped in the
#	training data.
#	input_size = frames * height * width * channels
#	size(frames) = [input_size, input_size]
print('Loading data...\n')
data_file = '../data/sprites/sprites_training.npz'
data = np.load(data_file)
input_frames = data['frames']
labels = data['labels']
actions = data['actions']

#########################################
###	  Setup arguments based on data   ###
#########################################
action_size = 1
num_frames = input_frames.shape[0]
num_input_channels = input_frames.shape[1]
num_output_channels = labels.shape[1]
input_height = input_frames.shape[2]
input_width = input_frames.shape[3]
num_input_frames = 3
input_size = num_input_frames * num_input_channels * input_height * input_width
hidden_size = 100

# Replace the dense weights (and bias vector) that take you from the action 
# 	vector to the vector that you point-wise prod with the encoded feature
#	vector.  Let's do this by setting all weights to 0 and a bias of 1 so 
#	we don't have to think about what we input with the actions.

# model will be the original action-conditiona model (with transformations)
# non-action-conditional model (without transformations)
print('Copying over model and re-setting weights in nac mode...')
nac_model = model_from_yaml(model.to_yaml())

zero_weights = np.zeros((action_size, hidden_size))
one_biases = np.ones((hidden_size,))
orig_weights = model.layers[action_projection_layer_index].get_weights()
orig_weights[0] = zero_weights
orig_weights[1] = one_biases
nac_model.layers[action_projection_layer_index].set_weights(orig_weights)

# Now display visualization
#	- true image
#	- action conditioned image
#	- non-transformed image

# Set up image display
plt.ion()
fig = plt.figure()
nac_handle = fig.add_subplot(1,3,1)
nac_handle.set_title('Non-action-conditional Image')
pred_handle = fig.add_subplot(1,3,2)
pred_handle.set_title('Action-conditional Image')
true_handle = fig.add_subplot(1,3,3)
true_handle.set_title('True Image')

display_iters = 1000

print('Displaying...')
for i in range(0, display_iters):
	# Input
	frame_input = input_frames[i,:,:,:]
	frame_input = np.reshape(frame_input, [1, num_input_channels, input_height, input_width])
	action_input = actions[i]
	action_input = np.reshape(actions[i], [1, action_size])

	# Non-action-conditioned predicted image
	prediction_input = [frame_input, action_input]		
	nac_predicted_frame = nac_model.predict(prediction_input)
	nac_predicted_frame = np.reshape(nac_predicted_frame, [input_height, input_width])
	nac_predicted_frame = nac_predicted_frame * SCALE_FACTOR * 100
	print(str(nac_predicted_frame))
	print()

	# Action-Conditioned Predicted  Image
	prediction_input = [frame_input, action_input]		
	ac_predicted_frame = model.predict(prediction_input)
	ac_predicted_frame = np.reshape(ac_predicted_frame, [input_height, input_width])
	ac_predicted_frame = ac_predicted_frame * SCALE_FACTOR
	print(str(ac_predicted_frame))
	print()

	# Compare to true image
	true_frame = labels[i,:,:,:]
	true_frame = np.reshape(true_frame, [input_height, input_width])
	true_frame = true_frame * SCALE_FACTOR
	print(str(true_frame))
	print()

	# Display
	nac_handle = fig.add_subplot(1,3,1)
	pred_handle = fig.add_subplot(1,3,2)
	true_handle = fig.add_subplot(1,3,3)

	if i == 0:
		nhandle = nac_handle.imshow(nac_predicted_frame)
		phandle = pred_handle.imshow(ac_predicted_frame)
		thandle = true_handle.imshow(true_frame)
	else:
		nhandle.set_data(nac_predicted_frame)
		phandle.set_data(ac_predicted_frame)
		thandle.set_data(true_frame)

	fig.show()
	ans = raw_input("Press [enter] to continue, type anything to exit.")
	if len(ans) > 0:
		sys.exit(0)


