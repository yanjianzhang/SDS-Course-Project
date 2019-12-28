import random



def threatMain(state, boardLen=20, reverse=False):
	random.seed(a=4, version=2)
	if reverse:
		for i in range(len(state)):
			for j in range(len(state[0])):
				if state[i][j] == 1:
					state[i][j] = 2
				elif state[i][j] == 2:
					state[i][j] = 1
	# if not reverse:
	# trd_pattern_list = ['211110', '011112', '01110', '2011100', '0011102', '010110', '011010', '011110', '11111']
	trd_pattern_list = ['011112', '01110', '010110', '011010', '11101', '10111', '11011', '211110']
	# else:
	# 	trd_pattern_list = ['011110', '11111']




	rest_square = findRestSquare(state, boardLen,  reverse)

	# print("origin resqure", rest_square)
	if not rest_square:
		high_prior_action, prior = findSure(state, boardLen, reverse)
		if prior:
			return high_prior_action, prior
		else:
			return None, None
	random.shuffle(rest_square)
	# calculate num of each rest
	rest_square_dict = {}
	rest_piority = {}
	for rest in rest_square:
		if rest[0] not in rest_piority:
			rest_piority[rest[0]] = rest[2]
		else:
			if rest[2] > rest_piority[rest[0]]:
				rest_piority[rest[0]] = rest[2]
		if rest[0] not in rest_square_dict:
			# initial
			rest_square_dict[rest[0]] = [rest[1]]
		else:
			# distinct the direction
			if rest[1] not in rest_square_dict[rest[0]]:
				rest_square_dict[rest[0]].append(rest[1])
	# count the number
	for i in rest_square_dict:
		rest_square_dict[i] = (rest_piority[i], len(rest_square_dict[i]))
	rest_tuple = sorted(rest_square_dict.items(), key=lambda d: (d[1][0], d[1][1]) if d[1][1] >= 2 else (-999, 999), reverse=True)
	# >=2, rest overlap, keep dependency
	high_prior_action, prior = findSure(state, boardLen, reverse)
	# print("prior",reverse,high_prior_action,prior)


	point_rest_tuple = [x[0] for x in rest_tuple]
	# print("the sure one", set(high_prior_action) & set(point_rest_tuple), prior)
	# print(high_prior_action, "h", rest_tuple)
	if prior:
		if isinstance(high_prior_action,list) and set(high_prior_action)&set(point_rest_tuple):
			return list(set(high_prior_action)&set(point_rest_tuple))[0], prior
		return high_prior_action, prior

	# print(point_rest_tuple)
	# print("thus",rest_tuple[0][0])
	# temp = []
	# for index in range(len(rest_square)):
	# 	temp.append(rest_square[index][0])
	# rest_square = temp
	# print("restuple",rest_tuple)
	if rest_tuple[0][1][1] >= 2:
		# print("in")
		return rest_tuple[0][0], 3 - rest_tuple[0][1][0]/5.0

	rest_level_3_5 = []
	if len(rest_square) >= 3:
		# combination
		for i in range(len(rest_square)):
			# if reverse: break
			for j in range(i):
				for k in range(j):
					pair = [rest_square[i], rest_square[j], rest_square[k]]
					if pair[1][0] == pair[2][0] or pair[1][0] == pair[0][0] or pair [0][0] == pair[2][0]:
						continue
					# print(pair)
					# horizon
					if pair[0][0][0] == pair[1][0][0] and pair[1][0][0] == pair[2][0][0] and pair[0][1] != 'px' and pair[1][1] != 'px' and pair[2][1] != 'px':
						state_str = []
						for chess in state[pair[0][0][0]]:
							state_str.append(str(chess))
						state_str[pair[0][0][1]] = '1'
						state_str[pair[1][0][1]] = '1'
						state_str[pair[2][0][1]] = '1'
						for trd in trd_pattern_list:
							if trd in "".join(state_str):
								trd_index = findIndex("".join(state_str), trd)
								for index_item in trd_index:
									if index_item <= min(pair[0][0][1], pair[1][0][1], pair[2][0][1]) and \
										index_item + len(trd) >= max(pair[0][0][1], pair[1][0][1], pair[2][0][1]):
										rest_level_3_5.append(random.choice(pair)[0])
					# vertical
					if pair[0][0][1] == pair[1][0][1] and pair[0][0][1] == pair[2][0][1] and pair[0][1] != 'cz' and pair[1][1] != 'cz' and pair[2][1] != 'cz':
						state_str = []
						for chess in state[pair[0][0][1]]:
							state_str.append(str(chess))
						state_str[pair[0][0][0]] = '1'
						state_str[pair[1][0][0]] = '1'
						state_str[pair[2][0][0]] = '1'
						for trd in trd_pattern_list:
							if trd in "".join(state_str):
								trd_index = findIndex("".join(state_str), trd)
								for index_item in trd_index:
									if index_item <= min(pair[0][0][0], pair[1][0][0], pair[2][0][0]) and \
										index_item + len(trd) >= max(pair[0][0][0], pair[1][0][0], pair[2][0][0]):
										rest_level_3_5.append(random.choice(pair)[0])
					# diagonal
					if pair[0][0][1] - pair[1][0][1] == pair[0][0][0] - pair[1][0][0] and \
							pair[0][0][1] - pair[2][0][1] == pair[0][0][0] - pair[2][0][0] and \
							pair[0][1] != 'yx' and pair[1][1] != 'yx' and pair[2][1] != 'yx':
						if pair[0][0][0] <= pair[0][0][1]:
							begin = pair[0][0][1] - pair[0][0][0]
							state_str = []
							for chess in range(boardLen):
								try:
									state_str.append(str(state[chess][begin + chess]))
								except:
									pass
							state_str[pair[0][0][0]] = '1'
							state_str[pair[1][0][0]] = '1'
							state_str[pair[2][0][0]] = '1'
							for trd in trd_pattern_list:
								if trd in "".join(state_str):
									trd_index = findIndex("".join(state_str), trd)
									for index_item in trd_index:
										if index_item <= min(pair[0][0][0], pair[1][0][0], pair[2][0][0]) and \
											index_item + len(trd) <= max(pair[0][0][0], pair[1][0][0], pair[2][0][0]):
											rest_level_3_5.append(random.choice(pair)[0])
						if pair[0][0][0] > pair[0][0][1]:
							begin = pair[0][0][0] - pair[0][0][1]
							state_str = []
							for chess in range(boardLen):
								try:
									state_str.append(str(state[begin + chess][chess]))
								except:
									pass
							state_str[pair[0][0][1]] = '1'
							state_str[pair[1][0][1]] = '1'
							state_str[pair[2][0][1]] = '1'
							for trd in trd_pattern_list:
								if trd in "".join(state_str):
									trd_index = findIndex("".join(state_str), trd)
									for index_item in trd_index:
										if index_item <= min(pair[0][0][1], pair[1][0][1], pair[2][0][1]) and \
											index_item + len(trd) <= max(pair[0][0][1], pair[1][0][1], pair[2][0][1]):
											rest_level_3_5.append(random.choice(pair)[0])
					# counter-diagonal
					if pair[0][0][1] - pair[1][0][1] == pair[1][0][0] - pair[0][0][0] and \
							pair[0][0][1] - pair[2][0][1] == pair[2][0][0] - pair[0][0][0] and \
							pair[0][1] != 'zx' and pair[1][1] != 'zx' and pair[2][1] != 'zx':
						if pair[0][0][0] + pair[0][0][1] <= boardLen - 1:
							begin = pair[0][0][0] + pair[0][0][1]
							state_str = []
							for chess in range(boardLen):
								try:
									state_str.append(str(state[chess][begin - chess]))
								except:
									pass
							state_str[pair[0][0][0]] = '1'
							state_str[pair[1][0][0]] = '1'
							state_str[pair[2][0][0]] = '1'
							for trd in trd_pattern_list:
								if trd in "".join(state_str):
									trd_index = findIndex("".join(state_str), trd)
									for index_item in trd_index:
										if index_item <= min(pair[0][0][0], pair[1][0][0], pair[2][0][0]) and \
											index_item + len(trd) <= max(pair[0][0][0], pair[1][0][0], pair[2][0][0]):
											rest_level_3_5.append(random.choice(pair)[0])
						if pair[0][0][0] + pair[0][0][1] > boardLen - 1:
							begin = pair[0][0][0] + pair[0][0][1]
							state_str = []
							for chess in range(boardLen):
								try:
									state_str.append(str(state[begin - chess][chess]))
								except:
									pass
							state_str[boardLen - 1 - pair[0][0][1]] = '1'
							state_str[boardLen - 1 - pair[1][0][1]] = '1'
							state_str[boardLen - 1 - pair[2][0][1]] = '1'
							for trd in trd_pattern_list:
								if trd in "".join(state_str):
									trd_index = findIndex("".join(state_str), trd)
									for index_item in trd_index:
										if index_item <= min(boardLen - 1 - pair[0][0][1],
										                                        boardLen - 1 - pair[1][0][1],
										                                        boardLen - 1 - pair[2][0][1]) and \
											index_item + len(trd) <= max(boardLen - 1 - pair[0][0][1],
											                                                boardLen - 1 - pair[1][0][1],
											                                                boardLen - 1 - pair[2][0][1]):
											rest_level_3_5.append(random.choice(pair)[0])

	rest_level_4 = []
	if len(rest_square) >= 2:
		# combination
		for i in range(len(rest_square)):
			# if reverse: break
			for j in range(i):
				pair = [rest_square[i], rest_square[j]]
				if pair[0][0] == pair[1][0]:
					continue
				# print(pair)
				# horizon
				if pair[0][0][0] == pair[1][0][0] and pair[0][1] != 'px' and pair[1][1] != 'px':
					state_str = []
					for chess in state[pair[0][0][0]]:
						state_str.append(str(chess))
					state_str[pair[0][0][1]] = '1'
					state_str[pair[1][0][1]] = '1'
					for trd in trd_pattern_list:
						if trd in "".join(state_str):
							trd_index = findIndex("".join(state_str), trd)
							for index_item in trd_index:
								if index_item <= min(pair[0][0][1], pair[1][0][1]) and \
									index_item + len(trd) >= max(pair[0][0][1], pair[1][0][1]):
									rest_level_4.append(random.choice(pair)[0])
				# vertical
				if pair[0][0][1] == pair[1][0][1] and pair[0][1] != 'cz' and pair[1][1] != 'cz':
					state_str = []
					for chess in state[pair[0][0][1]]:
						state_str.append(str(chess))
					state_str[pair[0][0][0]] = '1'
					state_str[pair[1][0][0]] = '1'
					for trd in trd_pattern_list:
						if trd in "".join(state_str):
							trd_index = findIndex("".join(state_str), trd)
							for index_item in trd_index:
								if index_item<= min(pair[0][0][0], pair[1][0][0]) and \
									index_item + len(trd) >= max(pair[0][0][0], pair[1][0][0]):
									rest_level_4.append(random.choice(pair)[0])
				# diagonal
				if pair[0][0][1] - pair[1][0][1] == pair[0][0][0] - pair[1][0][0] and pair[0][1] != 'yx' and pair[1][1] != 'yx':
					if pair[0][0][0] <= pair[0][0][1]:
						begin = pair[0][0][1] - pair[0][0][0]
						state_str = []
						for chess in range(boardLen):
							try:
								state_str.append(str(state[chess][begin + chess]))
							except:
								pass
						state_str[pair[0][0][0]] = '1'
						state_str[pair[1][0][0]] = '1'
						for trd in trd_pattern_list:
							if trd in "".join(state_str):
								trd_index = findIndex("".join(state_str), trd)
								for index_item in trd_index:
									if index_item <= min(pair[0][0][0], pair[1][0][0]) and \
										index_item + len(trd) <= max(pair[0][0][0], pair[1][0][0]):
										rest_level_4.append(random.choice(pair)[0])
					if pair[0][0][0] > pair[0][0][1]:
						begin = pair[0][0][0] - pair[0][0][1]
						state_str = []
						for chess in range(boardLen):
							try:
								state_str.append(str(state[begin + chess][chess]))
							except:
								pass
						state_str[pair[0][0][1]] = '1'
						state_str[pair[1][0][1]] = '1'
						for trd in trd_pattern_list:
							if trd in "".join(state_str):
								trd_index = findIndex("".join(state_str), trd)
								for index_item in trd_index:
									if index_item <= min(pair[0][0][1], pair[1][0][1]) and \
										index_item + len(trd) <= max(pair[0][0][1], pair[1][0][1]):
										rest_level_4.append(random.choice(pair)[0])
				# counter-diagonal
				if pair[0][0][1] - pair[1][0][1] == pair[1][0][0] - pair[0][0][0] and pair[0][1] != 'zx' and pair[1][1] != 'zx':
					if pair[0][0][0] + pair[0][0][1] <= boardLen - 1:
						begin = pair[0][0][0] + pair[0][0][1]
						state_str = []
						for chess in range(boardLen):
							try:
								state_str.append(str(state[chess][begin - chess]))
							except:
								pass
						state_str[pair[0][0][0]] = '1'
						state_str[pair[1][0][0]] = '1'
						for trd in trd_pattern_list:
							if trd in "".join(state_str):
								trd_index = findIndex("".join(state_str), trd)
								for index_item in trd_index:
									if index_item <= min(pair[0][0][0], pair[1][0][0]) and \
										index_item + len(trd) <= max(pair[0][0][0], pair[1][0][0]):
										rest_level_4.append(random.choice(pair)[0])
					if pair[0][0][0] + pair[0][0][1] > boardLen - 1:
						begin = pair[0][0][0] + pair[0][0][1]
						state_str = []
						for chess in range(boardLen):
							try:
								state_str.append(str(state[begin - chess][chess]))
							except:
								pass
						state_str[boardLen - 1 - pair[0][0][1]] = '1'
						state_str[boardLen - 1 - pair[1][0][1]] = '1'
						for trd in trd_pattern_list:
							if trd in "".join(state_str):
								trd_index = findIndex("".join(state_str), trd)
								for index_item in trd_index:
									if index_item <= min(boardLen - 1 - pair[0][0][1],
									                                        boardLen - 1 - pair[1][0][1]) and \
										index_item + len(trd) <= max(boardLen - 1 - pair[0][0][1],
										                                                boardLen - 1 - pair[1][0][1]):
										rest_level_4.append(random.choice(pair)[0])

	if rest_level_4:
		return random.choice(rest_level_4), 4
	elif rest_level_3_5:
		return random.choice(rest_level_3_5), 4

	# if reverse: return None, None
	#final_choice = sorted(rest_square, key=lambda x: rest_piority[x])[0]
	temp = []
	for index in range(len(rest_square)):
		temp.append(rest_square[index][0])
	rest_square = sorted(temp, key=lambda x: rest_piority[x], reverse=True)
	if not rest_square:
		return None

	new_rs = list(set(rest_square))
	if len(new_rs) >= 3:
		for i in range(len(new_rs)):
			for j in range(i):
				for k in range(j):
					if (mhtDistance(new_rs[i], new_rs[j]) <= 1.5 and mhtDistance(new_rs[i],
					                                                             new_rs[k]) <= 1.5) or \
						(mhtDistance(new_rs[k], new_rs[j]) <= 1.5 and mhtDistance(new_rs[k],
						                                                            new_rs[i]) <= 1.5) or \
						(mhtDistance(new_rs[j], new_rs[k]) <= 1.5 and mhtDistance(new_rs[j],
						                                                          new_rs[i]) <= 1.5):
						return [new_rs[i], new_rs[j], new_rs[k]], 4.5
	return rest_square, 5

def mhtDistance(cord1, cord2):
   return ((cord1[0] - cord2[0])**2 + (cord1[1] - cord2[1])**2)**0.5

# return random.choice(rest_square)




# when '011110' or '11111' are found return it
def findSure(board, boardLen=20, reverse=False):
	rest_square = []
	trd_pattern_list = ['11111','011110']
	threat_state = [[2 for i in range(boardLen + 2)] for j in range(boardLen + 2)]
	# print("the whole table", threat_state, boardLen)
	for i in range(boardLen):
		for j in range(boardLen):
			threat_state[i + 1][j + 1] = board[i][j]
	# print("the whole table",threat_state)
	for enum_index, pattern_len in enumerate([5, 6]):
		currentResult = []
		for i in range(boardLen + 2):
			for j in range(boardLen + 2):
				px = ''  # horizon
				cz = ''  # vertical
				zx = ''  # counter-diagonal
				yx = ''  # diagonal
				for temp in range(pattern_len):
					try:
						px += str(threat_state[i][j + temp])
					except:
						px = ''

					try:
						cz += str(threat_state[i + temp][j])
					except:
						cz = ''

					try:
						zx += str(threat_state[i + temp][(j - temp) if j - temp >= 0 else 9999])
					except:
						zx = ''

					try:
						yx += str(threat_state[i + temp][j + temp])
					except:
						yx = ''
				# 11111 is the first choice, 011110 is the second choice

				if px != '':  # horizon

					px = list(px)
					for index in range(pattern_len):
						if px[index] == '0':
							px[index] = '1'

							if "".join(px) in trd_pattern_list:
								currentResult.append(
									(i - 1, j + index - 1))  # "".join(px) == '11111' or '011110'
							px[index] = '0'
				if cz != '':  # vertical
					cz = list(cz)
					for index in range(pattern_len):
						if cz[index] == '0':
							cz[index] = '1'
							if "".join(cz) in trd_pattern_list:
								currentResult.append((i + index - 1, j - 1))
							cz[index] = '0'
				if zx != '':  # counter-diagonal
					zx = list(zx)
					for index in range(pattern_len):
						if zx[index] == '0':
							zx[index] = '1'
							if "".join(zx) in trd_pattern_list:
								currentResult.append((i + index - 1, j - index - 1))
							zx[index] = '0'
				if yx != '':  # diagonal
					yx = list(yx)
					for index in range(pattern_len):
						if yx[index] == '0':
							yx[index] = '1'
							if "".join(yx) in trd_pattern_list:
								currentResult.append((i + index - 1, j + index - 1))
							yx[index] = '0'
		if currentResult:
			return currentResult, enum_index + 1
	return None, None




def findRestSquare(board, boardLen=20, reverse=False):
	# print("the boardLen", boardLen)
	rest_square = []
	trd_pattern_list = {'011112': 4.2, '011100': 3.5, '001110': 3.5, '010110': 3.0, '011010': 3.0, '11101': 4, '10111': 4, '11011': 4, '211110': 4.2}
	threat_state = [[2 for i in range(boardLen + 2)] for j in range(boardLen + 2)]
	for i in range(boardLen):
		for j in range(boardLen):
			threat_state[i + 1][j + 1] = board[i][j]

	for i in range(boardLen + 2):
		for j in range(boardLen + 2):
			for pattern_len in [5, 6, 7]:
				px = ''
				cz = ''
				zx = ''
				yx = ''
				for temp in range(pattern_len):
					try:
						px += str(threat_state[i][j + temp])
					except:
						px = ''

					try:
						cz += str(threat_state[i + temp][j])
					except:
						cz = ''

					try:
						zx += str(threat_state[i + temp][(j - temp) if j - temp >= 0 else 9999])
					except:
						zx = ''

					try:
						yx += str(threat_state[i + temp][j + temp])
					except:
						yx = ''

				if px != '':
					px = list(px)
					for index in range(pattern_len):
						if px[index] == '0':
							px[index] = '1'
							if "".join(px) in trd_pattern_list:
								# item : action, direction
								rest_square.append([(i - 1, j + index - 1), 'px', trd_pattern_list["".join(px)]])
							px[index] = '0'
				if cz != '':
					cz = list(cz)
					for index in range(pattern_len):
						if cz[index] == '0':
							cz[index] = '1'
							if "".join(cz) in trd_pattern_list:
								rest_square.append([(i + index - 1, j - 1), 'cz', trd_pattern_list["".join(cz)]])
							cz[index] = '0'
				if zx != '':
					zx = list(zx)
					for index in range(pattern_len):
						if zx[index] == '0':
							zx[index] = '1'
							if "".join(zx) in trd_pattern_list:
								rest_square.append([(i + index - 1, j - index - 1), 'zx', trd_pattern_list["".join(zx)]])
							zx[index] = '0'
				if yx != '':
					yx = list(yx)
					for index in range(pattern_len):
						if yx[index] == '0':
							yx[index] = '1'
							if "".join(yx) in trd_pattern_list:
								rest_square.append([(i + index - 1, j + index - 1), 'yx', trd_pattern_list["".join(yx)]])
							yx[index] = '0'
	return rest_square

# find index of one trd in a sequence in combination part
def findIndex(str_sequence, mode):
	indexlist = []
	sequence = str_sequence
	while True:
		try:
			index = sequence.index(mode)
			indexlist.append(index)
			str_sequence = list(sequence)
			sequence[index] = '5'
			str_sequence = "".join(sequence)
		except:
			break
	return  indexlist
#
#
# boardLen = 20
# trd_pattern_list = ['211110', '011112', '01110', '2011100', '0011102', '010110', '011010', '011110']
# tables = [[[2 for i in range(boardLen + 2)] for j in range(boardLen + 2)] for k in range(len(trd_pattern_list))]
# for index, pattern in enumerate(trd_pattern_list):
# 	tables[index][2][1:len(pattern) + 1] = [int(i) for i in pattern]
# tables[3][1][2] = 0
# tables[3][3][2] = 1
# tables[3][4][2] = 1
# tables[3][5][2] = 1
# print(tables[3])
#
# print([threatMain(tables[k], boardLen) for k in range(len(trd_pattern_list))])
