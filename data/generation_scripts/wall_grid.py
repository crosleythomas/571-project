from empty_grid import EmptyGrid
import numpy as np
import random
import math

# Includes random walls in the grid

class WallGrid(EmptyGrid):
	WALL = 3
	def __init__(self, window_size, map_size):
		super(WallGrid, self).__init__(window_size, map_size)
		self.boundaries.append(self.WALL)
		self.grid.fill(self.WALL)
		self.explore((map_size / 2, map_size / 2), set())
		self.grid[self.pos] = self.SPRITE
		if window_size > 0:
			self.grid = np.lib.pad(self.grid, window_size / 2, self.pad_values)
		else:
			self.grid = np.lib.pad(self.grid, 1, self.pad_values)
		self.window_start = (self.window_start[0] + window_size / 2, self.window_start[1] + window_size / 2)
		self.pos = (self.pos[0] + window_size / 2, self.pos[1] + window_size / 2)


	def explore(self, current_pos, visited):
		if not self.in_bounds(current_pos):
			return None
		if current_pos in visited:
			return None
		visited.add(current_pos)
		if not self.would_complete_square(current_pos):
			self.grid[current_pos[0]][current_pos[1]] = self.PATH
			non_diag_options = self.non_diag_neighbors(current_pos)
			random.shuffle(non_diag_options)
			for cell in non_diag_options:
				self.explore(cell, visited)	

	def non_diag_neighbors(self, pos):
		top = (pos[0] - 1, pos[1])
		left = (pos[0], pos[1] - 1)
		bottom = (pos[0] + 1, pos[1])
		right = (pos[0], pos[1] + 1)
		return [top, left, bottom, right]

	def would_complete_square(self, pos):
		(top, left, bottom, right) = self.non_diag_neighbors(pos)
		top_left = [self.grid[pair] for pair in [top, left, (top[0], top[1] - 1)] if self.in_bounds(pair)]
		top_right = [self.grid[pair] for pair in [top, right, (top[0], top[1] + 1)] if self.in_bounds(pair)]
		bottom_left = [self.grid[pair] for pair in [bottom, left, (bottom[0], bottom[1] - 1)] if self.in_bounds(pair)]
		bottom_right = [self.grid[pair] for pair in [bottom, right, (bottom[0], bottom[1] + 1)] if self.in_bounds(pair)]
		return (sum(top_left) == 0  or sum(top_right) == 0  or 
				sum(bottom_left) == 0  or sum(bottom_right) == 0)

	def take_action(self, choice):
		super(WallGrid, self).take_action(choice)
		self.window_start = (self.pos[0] - self.window_size / 2, self.pos[1] - self.window_size / 2)
		
	def pad_values(self, vector, pad_width, iaxis, kwargs):
		vector[:pad_width[0]] = self.WALL
		vector[-pad_width[1]:] = self.WALL
		return vector

	def get_sprite(self, number):
		value = super(WallGrid, self).get_sprite(number)
		if value is not None:
			return value
		if number == self.WALL:
			return self.WALL
