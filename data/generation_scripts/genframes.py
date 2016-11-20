import scipy.io as sio
import numpy as np
import random
from tempfile import TemporaryFile

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

field_size = 1
num_channels = 1
side_length = sprite_width * (field_size * 2 + 1)
num_frames = 10000
frames = np.zeros((num_frames, num_channels, side_length, side_length), dtype=np.float64)

# initially in center
x = sprite_width * field_size
y = sprite_height * field_size
frames[0, :, y:y+sprite_height, x:x+sprite_width] = sprite

possible_actions = ['u', 'l', 'd', 'r', 'n']
action_indices = {'u' : 0, 'l' : 1, 'd' : 2, 'r': 3, 'n' : 4}

old_y = y
old_x = x

# Say first frame is initialized with no move to keep tensor sizes the same
# for frames and actions
actions = ['n']

for i in range(1, num_frames):
	choice = random.choice(possible_actions)
	if choice == 'u':
		y -= sprite_height
	elif choice == 'l':
		x -= sprite_width
	elif choice == 'd':
		y += sprite_height
	elif choice == 'r':
		x += sprite_width
	if x >= 0 and x < side_length and y >= 0 and y < side_length:
		old_y = y
		old_x = x
	else:
		x = old_x
		y = old_y
	frames[i, :, y:y+sprite_height, x:x+sprite_width] = sprite	
	actions.append(action_indices[choice])

if create_mat_file:
	sio.savemat("../sprites/sprites_baseline_data.mat", {'frames' : frames, 'actions' : actions })

if create_npz_file:
	np.savez('../sprites/sprites_baseline_data.npz', frames=frames, actions=actions)