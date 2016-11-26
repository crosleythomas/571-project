# The new gen frames - this should replace genframes.py but I made
# it a new file in case we want to maintain both

import scipy.io as sio
import numpy as np
import random
from tempfile import TemporaryFile
from wall_grid import WallGrid
from empty_grid import EmptyGrid
from dot_grid import DotGrid

create_mat_file = 1
create_npz_file = 0

# This file generates a file 'video.mat' with two values:
#	'alist' -> the image matrix, shaped (num_frames, width, height)
#	'actions' -> a char array with the actions
# len(actions) = len(alist) - 1 because the actions go between frames

random.seed(10)
	
sprite = sio.loadmat('../8x8_sprite.mat')['sprite']

sprite_width = sprite.shape[0]
sprite_height = sprite.shape[1]

num_channels = 1
window_size = 5
grid_size = 5
side_length = sprite_width * window_size
grid = WallGrid(window_size, grid_size) # can also be emptygrid

num_frames = 10000
frames = np.zeros((num_frames, num_channels, side_length, side_length), dtype=np.float64)
frames[0, :, :, :] = grid.convert_to_image()

possible_actions = ['u', 'l', 'd', 'r', 'n']
action_indices = {'u' : 0, 'l' : 1, 'd' : 2, 'r': 3, 'n' : 4}

# Say first frame is initialized with no move to keep tensor sizes the same
# for frames and actions
actions = []

for i in range(1, num_frames):
	choice = random.choice(possible_actions)
	grid.take_action(choice)
	frames[i, :, :, :] = grid.convert_to_image()
	actions.append(action_indices[choice])

if create_mat_file:
	sio.savemat("../sprites/sprites_baseline_data_dots.mat", {'frames' : frames, 'actions' : actions })

if create_npz_file:
	np.savez('../sprites/sprites_baseline_data.npz', frames=frames, actions=actions)