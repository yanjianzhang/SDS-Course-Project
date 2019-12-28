import random
import pisqpipe as pp
import copy
from pisqpipe import DEBUG_EVAL, DEBUG
from ADP import ADP
import network
from logUtil import *
from numpy import save
from numpy import load
from threat import threatMain

MAX_BOARD = 20
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]
isAround = [[False for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]
## for debug
# isAround[int(20 / 2) - 1][int(20 / 2) - 1] = True
# isAround[int(20 / 2)][int(20 / 2) + 1] = True
# isAround[5][8] = True
# print(isAround)
isStarted = True
epsilon = 0.05
gamma = 0.5
firstPlayer = "10"
activePlayer = "10"
win_pattern = "11111"
currentEmbed = None
nn = network.NeuralNetwork(input_nodes=218, hidden_nodes=60, output_nodes=1, learning_rate=0.05)


def actionNetwork():
	# given x_t, get action u_t
	resultAction = None
	selectable = []
	# if isStarted == True:

	threatAction, prior = threatMain(copy.deepcopy(board))
	logging.debug("our " + str(threatAction) + "|" + str(prior))
	enemy_threat_action, enemy_prior = threatMain(copy.deepcopy(board), reverse=True)
	# enemy_threat_action, enemy_prior = None, None
	logging.debug("enemy " + str(enemy_threat_action) + "|" + str(enemy_prior))

	skiprandom = False
	enemy = False
	if threatAction or enemy_threat_action:
		skiprandom = True
		# print("prior", threatAction, prior, enemy_threat_action, enemy_prior)

		if not enemy_threat_action or (threatAction and prior <= enemy_prior):
			resultAction = threatAction if not isinstance(threatAction, list) else max(threatAction, key=lambda x: criticNetworkEval(systemModel(x), getReward(x)))
			currentPrior = prior
		else:
			resultAction = enemy_threat_action if not isinstance(enemy_threat_action, list) else max(enemy_threat_action, key=lambda x: criticNetworkEval(systemModel(x), getReward(x)))
			currentPrior = enemy_prior
			# enemy = True
	else:
		updataAround()
		for i in range(MAX_BOARD):
			for j in range(MAX_BOARD):
				if isAround[i][j] == True and board[i][j] == 0:
					selectable.append((i, j))

		if len(selectable) == 1:
			resultAction = selectable[0]
		else:
			# print([criticNetworkEval(systemModel(x), getReward(x)) for x in selectable])
			resultAction = max(selectable, key=lambda x: criticNetworkEval(systemModel(x), getReward(x)))
	# else:
	# 	resultAction =  (int(MAX_BOARD / 2), int(MAX_BOARD / 2))

	if getReward(resultAction) == 1:
		currentPrior = 1
		skiprandom = True

	if not skiprandom and random.random() < epsilon:
		random_resultAction = random.choice(selectable)
		criticNetworkTrain(systemModel(random_resultAction), getReward(random_resultAction))
	else:
		if skiprandom == True:
			# criticNetworkTrain(systemModel(resultAction), 1)
			criticNetworkTrain(systemModel(resultAction), 1 / float(currentPrior) if currentPrior == 1 else 1 / (40 * float(currentPrior)) * (-1/float(20) if enemy else 1))
		else:
			criticNetworkTrain(systemModel(resultAction), getReward(resultAction))
	logging.debug("final " + str(resultAction))
	# print("final " + str(resultAction))
	return resultAction


def systemModel(action):
	# given action u_t, get x_t+1
	newboard = copy.deepcopy(board)
	global currentEmbed
	currentEmbed = ADP(board, firstPlayer, activePlayer).stateEmbedding()
	currentEmbed += [0 for i in range(action[0]-3, action[0]+4 ) for j in range(action[1]-3, action[1]+4)]
	newboard[action[0]][action[1]] = 1
	embed = ADP(newboard, firstPlayer, activePlayer).stateEmbedding()
	embed +=  [newboard[i][j] if 0<= i < len(newboard) and 0 <= j < len(newboard) else 2 for i in range(action[0]-3, action[0]+4 ) for j in range(action[1]-3, action[1]+4)]
	return embed


def getReward(action):
	# print(action)
	newboard = copy.deepcopy(board)
	newboard[action[0]][action[1]] = 1
	pattern_len = len(win_pattern)
	for row in range(len(board)):  # modified
		for col in range(MAX_BOARD - pattern_len + 1):
			line_5_px = ''
			line_5_cz = ''
			for i in range(pattern_len):
				line_5_px += str(newboard[row][col + i])
				line_5_cz += str(newboard[col + i][row])
			if line_5_px == win_pattern:
				return 1
	for row in range(MAX_BOARD):  # calculate the pattern in diagonal direction
		for col in range(MAX_BOARD):
			line_5_left = ''
			line_5_right = ''
			for i in range(pattern_len):
				if row + i <= MAX_BOARD - 1 and col - i >= 0:
					line_5_left += str(newboard[row + i][col - i])
				else:
					line_5_left = ''
				if row + i <= MAX_BOARD - 1 and col + i <= MAX_BOARD - 1:
					line_5_right += str(newboard[row + i][col + i])
				else:
					line_5_right = ''
			if line_5_left == win_pattern or line_5_right == win_pattern:  # modified
				return 1
	return 0


def criticNetworkEval(embed, reward):
	# given x_t, get V_t
	V = nn.run(embed)
	return V + reward


def criticNetworkTrain(embed2, reward):
	# given x_t, get V_t
	global gamma
	nn.train(currentEmbed, embed2, reward, gamma)
	return

def updataAround():
	isSetted = False
	for x in range(0, MAX_BOARD):
		for y in range(0, MAX_BOARD):
			isAround[x][y] = False
			for i in range(max(0, x - 1), min(MAX_BOARD, x + 2)):
				for j in range(max(0, y - 1), min(MAX_BOARD, y + 2)):
					if board[i][j] == 1 or board[i][j] == 2:
						isAround[x][y] = True
						isSetted = True
						break
				if isAround[x][y] == True:
						break
	if not isSetted:
		isAround[int(MAX_BOARD / 2) - 1][int(MAX_BOARD / 2) - 1] = True


# def setAround(x, y):
# 	noAround = True
# 	for i in range(0, MAX_BOARD):
# 		for j in range(0, MAX_BOARD):
# 			isAround[i][j] = False
# 	for i in range(max(0, x - 1), min(MAX_BOARD, x + 2)):
# 		for j in range(max(0, y - 1), min(MAX_BOARD, y + 2)):
# 			if board[i][j] == 0:
# 				isAround[i][j] = True
# 				noAround = False
# 	if noAround == True:
# 		isAround[int(20 / 2) - 1][int(20 / 2) - 1] = True
# 	isAround[x][y] = False


pp.infotext = 'name="pbrain-pyrandom", author="Jan Stransky", version="1.0", country="Czech Republic", www="https://github.com/stranskyjan/pbrain-pyrandom"'


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
		pass
	logging.debug("\ninitttttttttttttttttttt4 post" + str(nn.weights_hidden_to_output) + "\n5 post" + str(
		nn.weights_input_to_hidden))
	pp.pipeOut("OK")


def brain_restart():
	for x in range(pp.width):
		for y in range(pp.height):
			board[x][y] = 0

	pp.pipeOut("OK")


def isFree(x, y):
	# print(x >=0, y>=0, x < pp.width, y < pp.height, board[x][y])
	return x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] == 0


def brain_my(x, y):
	if isFree(x, y):
		board[x][y] = 1
		# setAround(x, y)
	else:
		pp.pipeOut("ERROR my move [{},{}]".format(x, y))


def brain_opponents(x, y):
	if isFree(x, y):
		board[x][y] = 2
		# setAround(x, y)
	else:
		pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))


def brain_block(x, y):
	if isFree(x, y):
		board[x][y] = 3
	else:
		pp.pipeOut("ERROR winning move [{},{}]".format(x, y))


def brain_takeback(x, y):
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


def brain_end():
	save("D:/SP2(资料存储室)/复旦学习资料/大三下/人工智能/Final/piskvork/model/input2hidden.model", nn.weights_input_to_hidden)
	save("D:/SP2(资料存储室)/复旦学习资料/大三下/人工智能/Final/piskvork/model/hidden2output.model", nn.weights_hidden_to_output)
	pass


def brain_about():
	pp.pipeOut(pp.infotext)


if DEBUG_EVAL:
	import win32gui


	def brain_eval(x, y):
		# TODO check if it works as expected
		wnd = win32gui.GetForegroundWindow()
		dc = win32gui.GetDC(wnd)
		rc = win32gui.GetClientRect(wnd)
		c = str(board[x][y])
		win32gui.ExtTextOut(dc, rc[2] - 15, 3, 0, None, c, ())
		win32gui.ReleaseDC(wnd, dc)

######################################################################
# A possible way how to debug brains.
# To test it, just "uncomment" it (delete enclosing """)
######################################################################
"""
# define a file for logging ...
DEBUG_LOGFILE = "/tmp/pbrain-pyrandom.log"
# ...and clear it initially
with open(DEBUG_LOGFILE,"w") as f:
	pass

# define a function for writing messages to the file
def logDebug(msg):
	with open(DEBUG_LOGFILE,"a") as f:
		f.write(msg+"\n")
		f.flush()

# define a function to get exception traceback
def logTraceBack():
	import traceback
	with open(DEBUG_LOGFILE,"a") as f:
		traceback.print_exc(file=f)
		f.flush()
	raise

# use logDebug wherever
# use try-except (with logTraceBack in except branch) to get exception info
# an example of problematic function
def brain_turn():
	logDebug("some message 1")
	try:
		logDebug("some message 2")
		1. / 0. # some code raising an exception
		logDebug("some message 3") # not logged, as it is after error
	except:
		logTraceBack()
"""
# test for reward
# newboard = board.copy()
# newboard[0][0:5] = [1,1,1,1,1]
# # print(getReward((4,5)))
# board = newboard
# pp.width = 20
# pp.height = 20
# actionNetwork()
# actionNetwork()
# actionNetwork()
# # actionNetwork()
# actionNetwork()
# actionNetwork()
# brain_turn()
# brain_opponents(3,4)
# actionNetwork()
# brain_turn()
# exit(88)

######################################################################

# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about
if DEBUG_EVAL:
	pp.brain_eval = brain_eval


def main():
	# print("here")
	pp.main()


if __name__ == "__main__":
	main()
