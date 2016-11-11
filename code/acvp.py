####################################################################
# Implementation of Oh, et al. action-conditoinal video prediction #
#	in keras.													   #
#																   #
# UW CSE 571 Project											   #
# Thomas Crosley and Karan Singh								   #
####################################################################

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Merge

###############################
###        Load data        ###
###############################

# Load frame data as numpy arrays
#	Input has to be arranged such that, even if the input for a single training
#	example has multiple concatenated frames, all frames are a grouped in the
#	training data.
#	input_size = frames * height * width * channels
#	size(frames) = [input_size, input_size]
frames = 

# Load action data
#	There are N timesteps, t = 0 is the initial step, t = i is the system at time i
#	range: actions[0]...actions[N-1], timesteps from [0...N-1]
#	actions[t] is a one-hot binary vector of actions
#	size(actions) = [timesteps, num_actions]
actions = 

#####################################
### Setup arguments based on data ###
#####################################
action_size = ( , 1)
inputs = 
labels = 
num_frames = 1
input_height = 
input_width = 
num_input_channels =
num_input_frames = 3
input_size = num_input_frames * frame_height * frame_width * num_input_channels
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
fmodel.add(Convolution2D(64, 3, 3, init='glorot_uniform', activation='relu', border_mode='same', input_shape=(input_channels, input_height, input_width)))
fmodel.add(Convolution2D(32, 3, 3, border_mode='same'))

# Add fully-connected layer(s) WITH non-linearity
fmodel.add(Dense(32, activation='relu'))

# Add fully-connected layer(s) WITHOUT non-linearity
fmodel.add(Dense(hidden_size, init='glorot_uniform'))

#####################################
### Step 2: Action transformation ###
#####################################

# Action multiplication
amodel.add(Dense(hidden_size, init='glorot_uniform'))

# Point-wise multiplicative combination
encoded = Merge([fmodel, amodel], mode='mul')

#############################
### Step 3: Deconvolution ###
#############################
# Deconvolutional Branch takes the merge branches as input
dbranch = Sequential()
dbranch.add(encoded)

# Add deconvolutional layer(s)
dbranch.add(Deconvolution2D(128, 4, 4, output_shape=(None, 3, 14, 14), border_mode='valid'))

######################
### Step 4: Train! ###
######################
dmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
dmodel.fit([frames, actions], labels)
