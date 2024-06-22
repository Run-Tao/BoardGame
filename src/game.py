class Board:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.winner = None

    def make_move(self, x, y, player):
        if self.board[x][y] == ' ':
            self.board[x][y] = player
            if self.check_winner(x, y, player):
                self.winner = player
            return True
        return False

    def check_winner(self, x, y, player):
        if all(self.board[x][i] == player for i in range(3)):
            return True
        if all(self.board[i][y] == player for i in range(3)):
            return True
        if x == y and all(self.board[i][i] == player for i in range(3)):
            return True
        if x + y == 2 and all(self.board[i][2 - i] == player for i in range(3)):
            return True
        return False

    def is_full(self):
        return all(self.board[x][y] != ' ' for x in range(3) for y in range(3))

    def __str__(self):
        return '\n'.join([' '.join(self.board[i]) for i in range(3)])


class NestedTicTacToe:
    def __init__(self):
        self.main_board = Board()
        self.boards = [[Board() for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.current_board = (0, 0)
        self.last_move = None
        self.winner = None

    def make_move(self, main_x, main_y, sub_x, sub_y):
        if not self.is_valid_move(main_x, main_y, sub_x, sub_y):
            return False

        if self.boards[main_x][main_y].make_move(sub_x, sub_y, self.current_player):
            if self.boards[main_x][main_y].winner:
                self.main_board.make_move(main_x, main_y, self.current_player)
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            self.current_board = (sub_x, sub_y)
            self.last_move = (main_x, main_y, sub_x, sub_y)
            if self.main_board.winner:
                self.winner = self.main_board.winner
            return True
        return False

    def is_valid_move(self, main_x, main_y, sub_x, sub_y):
        # 如果目标小棋盘已满且没有赢家，允许自由选择
        if self.boards[self.current_board[0]][self.current_board[1]].is_full():
            return True
        # 否则，只能在指定的小棋盘内落子
        return (main_x == self.current_board[0] and main_y == self.current_board[1]) and \
               self.boards[main_x][main_y].board[sub_x][sub_y] == ' '

    def check_winner(self):
        return self.main_board.winner

    def is_full(self):
        return self.main_board.is_full()

    def is_terminal(self):
        return self.winner is not None or self.is_full()

    def get_legal_actions(self):
        legal_actions = []
        # 检查当前目标小棋盘是否已满
        if not self.boards[self.current_board[0]][self.current_board[1]].is_full():
            for x in range(3):
                for y in range(3):
                    if self.boards[self.current_board[0]][self.current_board[1]].board[x][y] == ' ':
                        legal_actions.append((self.current_board[0], self.current_board[1], x, y))
        else:
            for main_x in range(3):
                for main_y in range(3):
                    if not self.boards[main_x][main_y].is_full():
                        for sub_x in range(3):
                            for sub_y in range(3):
                                if self.boards[main_x][main_y].board[sub_x][sub_y] == ' ':
                                    legal_actions.append((main_x, main_y, sub_x, sub_y))
        return legal_actions

    def get_random_legal_action(self):
        import random
        return random.choice(self.get_legal_actions())

    def apply_move(self, move):
        self.make_move(*move)

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def get_reward(self):
        if self.winner == 'X':
            return 1
        elif self.winner == 'O':
            return -1
        else:
            return 0

    def display(self):
        for i in range(3):
            for j in range(3):
                print(f'Board ({i}, {j}):')
                print(self.boards[i][j])
                print()
        print('Main Board:')
        print(self.main_board)
