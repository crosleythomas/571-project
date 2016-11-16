####################################################################
# Implementation of Oh, et al. action-conditoinal video prediction #
#	in keras.													   #
#																   #
# UW CSE 571 Project											   #
# Thomas Crosley and Karan Singh								   #
####################################################################

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Merge, Flatten, Deconvolution2D, Reshape
import scipy.io as sio
import numpy as np

###############################
###        Load data        ###
###############################

# Load frame data as numpy arrays
#	Input has to be arranged such that, even if the input for a single training
#	example has multiple concatenated frames, all frames are a grouped in the
#	training data.
#	input_size = frames * height * width * channels
#	size(frames) = [input_size, input_size]

# right now we just have the one sprite.mat
frames = np.array([sio.loadmat('../data/8x8_sprite.mat')['sprite']])
input_height = frames[0].shape[0]
input_width = frames[0].shape[1]
num_frames = frames.shape[0]
num_input_channels = 1 # how to come up with this dynamically?

# we need to resize frames per this example:
# "Architecture for learning image captions with a convnet and a Gated Recurrent Unit:"
#	https://keras.io/getting-started/sequential-model-guide/
# from that example:
# 	frames should be a numpy float array of shape (nb_samples, nb_channels=?, width, height).
frames = frames.reshape((num_frames, num_input_channels, input_width, input_height))

# Load action data
# placeholder for now
actions = np.array([1])

#####################################
### Setup arguments based on data ###
#####################################

# labels will be a full output image for us - but i can't get deconv working
#	so for now it's just a random vector that's the same size as our current model output (enc)
labels = np.random.random((1, 32)) # np.array([1])
hidden_size = 32

###################
### Build Model ###
###################

# Action Branch
amodel = Sequential()

# Frames Branch
fmodel = Sequential()

########################
### Step 1: Encoding ###
########################

# Add convolutional layer(s)
# Convolution2D(num_filters, conv_width)
fmodel.add(Convolution2D(64, 3, 3, init='glorot_uniform', activation='relu', border_mode='same', input_shape=(num_input_channels, input_height, input_width)))
fmodel.add(Convolution2D(32, 3, 3, border_mode='same'))

fmodel.add(Flatten())

# Add fully-connected layer(s) WITH non-linearity
fmodel.add(Dense(hidden_size, activation='relu'))

# Add fully-connected layer(s) WITHOUT non-linearity
fmodel.add(Dense(hidden_size, init='glorot_uniform'))

#####################################
### Step 2: Action transformation ###
#####################################

# Action multiplication
amodel.add(Dense(hidden_size, init='glorot_uniform', input_dim=1))

# Point-wise multiplicative combination
encoded = Merge([fmodel, amodel], mode='mul')

#############################
### Step 3: Deconvolution ###
#############################
# Deconvolutional Branch takes the merge branches as input
dbranch = Sequential()
dbranch.add(encoded)

#dbranch.add(Dense(64))
#dbranch.add(Dense(64, init='glorot_uniform', activation='relu'))
#dbranch.add(Reshape((1,8,8)))

# Add deconvolutional layer(s)
# can't figure out how to do this size-wise
# dbranch.add(Deconvolution2D(128, 4, 4, output_shape=(None, 3, 14, 14), border_mode='valid'))

######################
### Step 4: Train! ###
######################
dbranch.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
dbranch.fit([frames, actions], labels)
