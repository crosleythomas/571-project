# The new gen frames - this should replace genframes.py but I made
# it a new file in case we want to maintain both

import scipy.io as sio
import numpy as np
import random
from tempfile import TemporaryFile
from wall_grid import WallGrid
from empty_grid import EmptyGrid
from ball import Ball

from scipy.misc import toimage
# See the environment with
#	toimage(grid.convert_to_image(1)).show()
create_mat_file = 0
create_npz_file = 1

# This file generates a file 'video.mat' with two values:
#	'alist' -> the image matrix, shaped (num_frames, width, height)
#	'actions' -> a char array with the actions
# len(actions) = len(alist) - 1 because the actions go between frames

random.seed(10)

sprite = sio.loadmat('../8x8_sprite.mat')['sprite']

sprite_width = sprite.shape[0]
sprite_height = sprite.shape[1]
side_length = 43
num_channels = 1
window_size = 0
grid_size = side_length

grid = Ball(grid_size) # can also be emptygrid
# grid = EmptyGrid(window_size, grid_size)
# grid = DotGrid(window_size, grid_size)
shape = grid.get_shape()
side_length = shape[0]

toimage(grid.convert_to_image(1)).show()
small_size = (10, 30)

height = small_size[1] - small_size[0]
num_frames = 1000
frames = np.zeros((num_frames, num_channels, height, side_length), dtype=np.float64)
frames[0, :, :, :] = grid.convert_to_image(1)[small_size[0]:small_size[1], :]

possible_actions = ['u', 'd', 't']
action_indices = {'u' : 0, 'd' : 1, 't' : 2}

# Say first frame is initialized with no move to keep tensor sizes the same
# for frames and actions
actions = []
for i in range(1, num_frames):
	choice = random.choice(possible_actions)
	grid.take_action(choice)
	frames[i, :, :, :] = grid.convert_to_image(1)[small_size[0]:small_size[1], :]
	
	actions.append(action_indices[choice])

if create_mat_file:
	sio.savemat("../sprites/ball.mat", {'frames' : frames, 'actions' : actions })

if create_npz_file:
	np.savez('../sprites/ball.npz', frames=frames, actions=actions)