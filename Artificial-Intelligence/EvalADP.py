# from numpy import zeros

def zeros(Len):
	return [0 for i in range(Len)]


class ADP():
    def __init__(self, state, fisrtPlayer, activePlayer):
        # state: the state of board, fisrtPlayer: the player running the first step('10' or '01'), activePlayer: the next step player('10' or '01')
        self.state = state
        self.fisrtPlayer = fisrtPlayer
        self.activePlayer = activePlayer
        self.state_embed = zeros(5 * (len(state) - 1) * 2 + 1 * 2 + len(state) * 2 * 2 + 1 * 2)
        # 1. Take 20 patterns as example: 5*19*2 means we have 19 patterns for both 2 players whose number needs to be encode with 5 dim. Besides, a
        # five-in-a-row pattern means one of the two plays has won't the game, so its number can be encode with only 1 dim(0/1), calculated as 1*2(2 players).
        # 2. We concatence a 2-dim vector to each of the pattern embedding above to indicate whether it's player 1's turn(10)  or player 2's(01).
        # 3. At last we concatence a 2-dim vector to indicate who go first at the beginning of the game.

    def stateEmbedding(self):
        # get state_embed
        state = self.state
        pattern_list = ['11111', '011110',
        '011100', '001110', '011010', '010110', '11110', '01111', '11011', '10111', '11101', '001100', '001010', \
        '010100', '000100', '001000', '22222', '022220',
        '022200', '002220', '022020', '020220', '22220', '02222', '22022', '20222', '22202', '002200',
        '002020', '020200', '000200', '002000']
        pattern_num_dict = {}
        for i in pattern_list:
            pattern_num_dict[i] = 0
        for pattern_len in [5, 6]:
            for row in range(len(state)):  # calculate the pattern in horizonal and perpendicular direction
                for col in range(len(state) - pattern_len + 1):
                    line_5_px = ''
                    line_5_cz = ''
                    for i in range(pattern_len):
                        line_5_px += str(state[row][col + i])
                        line_5_cz += str(state[col + i][row])
                    if line_5_px in pattern_num_dict:
                        pattern_num_dict[line_5_px] += 1
                    if line_5_cz in pattern_num_dict:
                        pattern_num_dict[line_5_cz] += 1
            for row in range(len(state)):  # calculate the pattern in diagonal direction
                for col in range(len(state)):
                    line_5_left = ''
                    line_5_right = ''
                    for i in range(pattern_len):
                        if row + i <= len(state)-1 and col - i >= 0:
                            line_5_left += str(state[row + i][col - i])
                        else:
                            line_5_left = ''
                        if row + i <= len(state)-1 and col + i <= len(state)-1:
                            line_5_right += str(state[row + i][col + i])
                        else:
                            line_5_right = ''
                    if line_5_left in pattern_num_dict:
                        pattern_num_dict[line_5_left] += 1
                    if line_5_right in pattern_num_dict:
                        pattern_num_dict[line_5_right] += 1
        embed = []
        for pattern in pattern_list:
            pattern_num = pattern_num_dict[pattern]
            if pattern == '11111' or pattern == '22222':
                embed = embed + [pattern_num, int(self.activePlayer[0]), int(self.activePlayer[1])]
            else:
                if pattern_num >= 5:
                    embed = embed + [1, 1, 1, 1, (pattern_num - 4) / 2, int(self.activePlayer[0]), int(self.activePlayer[1])]
                else:
                    embed = embed + [1] * pattern_num + [0] * (5 - pattern_num) + [int(self.activePlayer[0]), int(self.activePlayer[1])]
        embed = embed + [int(self.fisrtPlayer[0]), int(self.fisrtPlayer[1])]
        return embed


## nn.forward() X
# testboard = [[1, 1, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
# adp = ADP(testboard, "10", "10")
# embed = adp.stateEmbedding()
# print(len(embed))
