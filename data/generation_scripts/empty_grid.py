from grid import Grid

class EmptyGrid(Grid):
	def safe(self, new_pos):
		return self.grid[new_pos] not in self.boundaries

	def take_action(self, choice):
		new_pos = self.pos
		if choice == 'u':
			new_pos = (new_pos[0] - 1, new_pos[1])
		elif choice == 'l':
			new_pos = (new_pos[0], new_pos[1] - 1)
		elif choice == 'd':
			new_pos = (new_pos[0] + 1, new_pos[1])
		elif choice == 'r':
			new_pos = (new_pos[0], new_pos[1] + 1)

		if self.in_bounds(new_pos) and self.safe(new_pos):
			self.grid[self.pos] = self.PATH
			self.pos = new_pos
			self.grid[self.pos] = self.SPRITE