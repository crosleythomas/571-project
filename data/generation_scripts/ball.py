from empty_grid import EmptyGrid
import numpy as np

class Ball(EmptyGrid):
	DOT = 100
	def __init__(self, map_size):
		super(Ball, self).__init__(0, map_size)
		self.grid = np.zeros((map_size, map_size), dtype=np.float64)
		self.SPRITE_SIZE = 1
		self.grid[self.pos[0], self.pos[1]] = self.DOT
		self.direction = 1 # -1 or 1
		self.speed = 1
		self.tick_count = 0

	def create_dot(sprite_size):
		dots_sprite = np.zeros((sprite_size, sprite_size), dtype=np.float64)
		center = (sprite_size / 2) - 1
		dots_sprite[center:center + 2, center:center + 2] = 0.5
		return dots_sprite

	def get_sprite_size(self):
		return self.SPRITE_SIZE

	def take_action(self, choice):
		new_pos = self.pos
		self.tick_count += 1
		if choice == 't':
			self.direction = self.direction * -1
		elif choice == 'u':
			self.speed = self.speed + 1
		elif choice == 'd':
			self.speed = self.speed - 1
		if self.tick_count % 5 == 0:
			self.speed -= 1
		if self.speed <= 1:
			self.speed = 1
		elif self.speed >= 10:
			self.speed = 10
		new_pos = (new_pos[0], new_pos[1] + self.speed * self.direction)
		if self.in_bounds(new_pos) and self.safe(new_pos):
			self.grid[self.pos] = self.PATH
			self.pos = new_pos
			self.grid[self.pos] = self.DOT
		else:
			self.direction = self.direction * -1

	def get_sprite(self, number):
		value = super(Ball, self).get_sprite(number)
		if value is not None:
			return value
		if number == self.DOT:
			dots_sprite = np.zeros((self.SPRITE_SIZE, self.SPRITE_SIZE), dtype=np.float64)
			center = (self.SPRITE_SIZE / 2) - 1
			dots_sprite[center:center + 2, center:center + 2] = 0.5
			return dots_sprite		