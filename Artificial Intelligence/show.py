# -*- coding: utf-8 -*-
"""
@create_time : 2019-06-15 17:17
@environment : python 3.6
@author      : zhangjiwen
@file        : show.py
"""
import random
import time
import argparse
from example import *


class PP:
	# zs: to simulate the pisqpipe package
	def __init__(self):
		self.height = 20
		self.width = 20
		self.info_timeout_turn = 15000
		self.terminateAI = None

	def pipeOut(self, what):
		print(what)

	def do_mymove(self, x, y):
		brain_my(x, y)
		self.pipeOut("{},{}".format(x, y))

	def do_oppmove(self, x, y):
		brain_opponents(x, y)
		self.pipeOut("{},{}".format(x, y))


class EPP:
	# zs: to simulate the pisqpipe package
	def __init__(self):
		self.height = 20
		self.width = 20
		self.info_timeout_turn = 15000
		self.terminateAI = None

	def pipeOut(self, what):
		brain_opponents(x, y)
		print(what)

	def do_mymove(self, x, y):
		self.pipeOut("{},{}".format(x, y))

	def do_oppmove(self, x, y):
		brain_my(x, y)
		self.pipeOut("{},{}".format(x, y))


pp = PP()
epp = PP()

parser = argparse.ArgumentParser(description='DC-GAN on PyTorch')
parser.add_argument('--time-limit', default=5.0,
                    help='max time for one step', type=float)
parser.add_argument('--board-size', default=20,
                    help='board size, assuming the board is a square', type=int)
parser.add_argument('--n-in-line', default=20,
                    help='n=5 is the standard Gomoku', type=int)
parser.add_argument('--detail', default=False,
                    help='print the Nodes with higher than 0.4 winning rates', type=bool)
parser.add_argument('--max-simulation', default=20,
                    help='max simulation for one Node', type=int)
parser.add_argument('--max-simulation-one-step', default=150,
                    help='max simulation for one step, used for truncated simulation', type=int)
args = parser.parse_args()

MAX_BOARD = args.board_size


# board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]


def brain_init():
	if pp.width < 5 or pp.height < 5:
		pp.pipeOut("ERROR size of the board")
		return
	if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
		pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
		return
	try:
		nn.weights_input_to_hidden = load("D:/SP2(资料存储室)/复旦学习资料/大三下/人工智能/Final/piskvork/model/input2hidden.model.npy")
		nn.weights_hidden_to_output = load("D:/SP2(资料存储室)/复旦学习资料/大三下/人工智能/Final/piskvork/model/hidden2output.model.npy")
	except:
		print("load fail")
		pass
	logging.debug("\ninitttttttttttttttttttt4 post" + str(nn.weights_hidden_to_output) + "\n5 post" + str(nn.weights_input_to_hidden))
	pp.pipeOut("OK")


def brain_restart():
	for x in range(pp.width):
		for y in range(pp.height):
			board[x][y] = 0

	pp.pipeOut("OK")


def isFree(x, y):
	"""whether (x, y) is available"""
	return x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] == 0


def brain_my(x, y):
	"""
	Myself take a move.
	"""
	if isFree(x, y):
		board[x][y] = 1
	else:
		pp.pipeOut("ERROR my move [{},{}]".format(x, y))


def brain_opponents(x, y):
	"""
	My opponent take a move.
	"""
	# print("in ", x, y, isFree(x, y))
	# print(x >= 0, y >= 0, x < pp.width, y < pp.height, board[x][y] == 0, pp.width, pp.height)

	if isFree(x, y):
		board[x][y] = 2
	else:
		pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))


def brain_block(x, y):
	"""
	Another state of board -- block.
	Maybe it means that this block is unreachable for both of me and my opponent.
	"""
	if isFree(x, y):
		board[x][y] = 3
	else:
		pp.pipeOut("ERROR winning move [{},{}]".format(x, y))


def brain_takeback(x, y):
	"""
	Take back a move. Something like backtrack.
	return 0 -- back succeed
	return 2 -- an error
	You are now cheating!
	"""
	if x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] != 0:
		board[x][y] = 0
		return 0
	return 2


def brain_turn():
	# modified
	if pp.terminateAI:
		return
	i = 0
	while True:
		x, y = actionNetwork()

		i += 1
		if pp.terminateAI:
			return
		if isFree(x, y):
			# setAround(x, y)
			break
	if i > 1:
		pp.pipeOut("DEBUG {} coordinates didn't hit an empty field".format(i))
	pp.do_mymove(x, y)


def brain_end(x, y):
	pass


def brain_about():
	pp.pipeOut(pp.infotext)


def brain_show():
	st = '  '
	for i in range(len(board[0])):
		if i > 9:
			st += str(i) + ' '
		else:
			st += ' ' + str(i) + ' '
	print(st)
	c = 0
	for row in board:
		if c > 9:
			print(c, end=' ')
		else:
			print('', c, end=' ')
		c += 1
		st = ''
		for ii in row:
			if ii == 1:
				st += 'O  '
			elif ii == 2:
				st += 'X  '
			else:
				st += '-  '
		print(st)


def brain_play():
	while 1:
		print('(if you want to quit, ENTER quit)')
		x = input("Your turn, please give a coordinate 'x y':")
		print()
		if x == 'quit':
			print('You quit.')
			return None
		x = x.split()
		try:
			# brain_turn()
			for x1, y1 in [(7,2),(9,3),(9,5),(10,3),(10,4),(11,5),(11,6),(11,8),(12,7),(13,6)]:
				brain_opponents(x1,y1)
			for x2, y2 in [(6,2),(8,2),(9,2),(10,1),(10,6),(10,9),(11,4),(11,7),(12,5),(13,8)]:
				brain_my(x2,y2)
			brain_opponents(int(x[0]), int(x[1]))
		except ValueError or IndexError:
			print('Invalid input!')
			continue
		break
	# brain_show()
	return 0


def main():
	brain_init()
	brain_show()

	while brain_play() is not None:
		brain_turn()
		brain_show()


if __name__ == "__main__":
	main()
