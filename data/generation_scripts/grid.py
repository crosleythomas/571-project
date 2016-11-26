import numpy as np
import scipy.io as sio

# A common class for grid generation
#	Extend this class to customize the grid

# Maintains an internal representation of the grid that is image-agnostic
# convert_to_image() will output an image, whereas get_frame() will return
#	the internal representation

class Grid(object):
	SPRITE_SIZE = 8

	# All grids have a sprite and a path
	SPRITE = 1
	PATH = 0
	def __init__(self, window_size, map_size):
		self.window_size = window_size
		self.map_size = map_size
		self.grid = np.zeros((map_size, map_size), dtype=np.float64)
		self.pos = (map_size / 2, map_size / 2)
		self.window_start = (self.pos[0] - self.window_size / 2, self.pos[1] - self.window_size / 2)
		self.grid[self.pos] = self.SPRITE
		self.boundaries = []
	
	# returns whether the given position is inside the grid
	def in_bounds(self, pos):
		return not (pos[0] < 0 or pos[0] >= self.grid.shape[0] or 
					pos[1] < 0 or pos[1] >= self.grid.shape[1])			

	# returns a representation of the grid state, where each entity
	#	is represented by a number
	def get_frame(self):
		window = (self.window_start, (self.window_size, self.window_size))
		return self.grid[window[0][0]:window[0][0]+window[1][0], window[0][1]:window[0][1]+window[1][1]]

	# Used internally to map the sprite number to the specific pixel values
	#	when creating an image
	def get_sprite(self, number):
		if number == self.SPRITE:
			return sio.loadmat('../8x8_sprite.mat')['sprite']
		elif number == self.PATH:
			return self.PATH

	# Converts the internal representation to an image
	def convert_to_image(self):
		sprites = {}
		sprite_size = self.SPRITE_SIZE
		this_window = self.get_frame()
		side_length = this_window.shape[0] * sprite_size
		image = np.zeros((side_length, side_length), dtype=np.float64)
		for i in range(0, side_length, sprite_size):
			for j in range(0, side_length, sprite_size):
				value = this_window[(i / sprite_size, j / sprite_size)]
				if value not in sprites:
					sprites[value] = self.get_sprite(value)
				image[i:i+sprite_size, j:j+sprite_size] = sprites[value]
		return image