import scipy.io as sio
import numpy as np
import random

# This file generates a file 'video.mat' with two values:
#	'alist' -> the image matrix, shaped (num_frames, width, height)
#	'actions' -> a char array with the actions
# len(actions) = len(alist) - 1 because the actions go between frames

random.seed(10)

sprite = sio.loadmat('../data/8x8_sprite.mat')['sprite']
sprite_width = sprite.shape[0]
sprite_height = sprite.shape[1]

field_size = 1
side_length = sprite_width * (field_size * 2 + 1)
iter_length = 10
result = np.zeros((iter_length, side_length, side_length), dtype=np.float64)

# initially in center
x = sprite_width * field_size
y = sprite_height * field_size
result[0, y:y+sprite_height, x:x+sprite_width] = sprite

possible_actions = ['u', 'l', 'd', 'r', 'n']

old_y = y
old_x = x

choices = []

for i in range(1, iter_length):
	choice = random.choice(possible_actions)
	if choice == 'u':
		y -= sprite_height
	elif choice == 'l':
		x -= sprite_width
	elif choice == 'd':
		y += sprite_height
	elif choice == 'r': # right
		x += sprite_width
	if x >= 0 and x < side_length and y >= 0 and y < side_length:
		old_y = y
		old_x = x
	else:
		x = old_x
		y = old_y
	result[i, y:y+sprite_height, x:x+sprite_width] = sprite	
	choices.append(choice)

sio.savemat("video.mat", {'alist' : result, 'actions' : choices })