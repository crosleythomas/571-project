from empty_grid import EmptyGrid
import numpy as np

class DotGrid(EmptyGrid):
	DOT = 4
	def __init__(self, window_size, map_size):
		super(DotGrid, self).__init__(window_size, map_size)
		for i in range(self.map_size):
			for j in range(self.map_size):
				if (i, j) != self.pos:
					self.grid[i, j] = self.DOT

	def create_dot(sprite_size):
		dots_sprite = np.zeros((sprite_size, sprite_size), dtype=np.float64)
		center = (sprite_size / 2) - 1
		dots_sprite[center:center + 2, center:center + 2] = 0.5
		return dots_sprite

	def get_sprite(self, number):
		value = super(DotGrid, self).get_sprite(number)
		if value is not None:
			return value
		if number == self.DOT:
			dots_sprite = np.zeros((self.SPRITE_SIZE, self.SPRITE_SIZE), dtype=np.float64)
			center = (self.SPRITE_SIZE / 2) - 1
			dots_sprite[center:center + 2, center:center + 2] = 0.5
			return dots_sprite		