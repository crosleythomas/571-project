import scipy.io as sio
import numpy as np
import random

sprite = sio.loadmat('../data/8x8_sprite.mat')['sprite']
sprite_width = sprite.shape[0]
sprite_height = sprite.shape[1]


field_size = 1
side_length = sprite_width * (field_size * 2 + 1)
#field = np.zeros((side_length, side_length),dtype=np.float64)
iter_length = 10
result = np.zeros((side_length, side_length, iter_length), dtype=np.float64)


# initially in center
x = sprite_width * field_size
y = sprite_height * field_size
result[y:y+sprite_height, x:x+sprite_width, 0] = sprite

# sio.savemat("test.mat", {'alist' : field })

possible_actions = ['u', 'l', 'd', 'r', 'n']

old_y = y
old_x = x


#sio.savemat("num0" + ".mat", {'alist' : field })
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
		# field[old_y:old_y+sprite_height, old_x:old_x+sprite_width] = 0
		# field[y:y+sprite_height, x:x+sprite_width] = sprite
		# result[i, old_y:old_y+sprite_height, old_x:old_x+sprite_width] = 0
			
		old_y = y
		old_x = x
	else:
		x = old_x
		y = old_y
	result[y:y+sprite_height, x:x+sprite_width, i] = sprite	
	choices.append(choice)
	#sio.savemat("num" + str(i) + "action-" + choice + ".mat", {'alist' : field, 'action' : choice })

sio.savemat("video.mat", {'alist' : result, 'actions' : choices })