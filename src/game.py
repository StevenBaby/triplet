from ast import Mod
import pickle
import os
import random
import json

import numpy as np
from tqdm import tqdm
from logger import logger

SPAN = 3
BLACK = 1
WHITE = -1
DRAW = 2
EMPTY = 0


class Model(object):

    @classmethod
    def get_turn(cls, board):
        black = len(np.argwhere(board == BLACK))
        white = len(np.argwhere(board == WHITE))
        if (black - white) not in (0, 1):
            raise Exception("invalid item %s", board)

        if (black - white) == 1:
            return 1
        else:
            return -1

    @classmethod
    def is_final(cls, board):
        results = []

        turn = cls.get_turn(board=board)

        for line in board:
            results.append(sum(line))

        results.append(sum([
            board[(0, 0)],
            board[(1, 1)],
            board[(2, 2)],
        ]))

        results.append(sum([
            board[(2, 0)],
            board[(1, 1)],
            board[(0, 2)],
        ]))

        T = board.transpose()

        for line in T:
            results.append(sum(line))

        if turn == BLACK:
            result = max(results)
        else:
            result = min(results)

        if abs(result) == SPAN:
            return turn

        if not cls.get_args(board):
            return DRAW

        return 0

    @classmethod
    def new_board(cls):
        board = np.zeros((3, 3), dtype=np.int8)
        return board

    @classmethod
    def get_hash(cls, board):
        result = 0
        for line in board:
            for var in line:
                assert (var in (0, 1, -1))
                result <<= 2
                result |= (int(var) + 2)
        return result

    @classmethod
    def get_args(cls, board):
        return [
            (arg[0], arg[1])
            for arg in np.argwhere(board == EMPTY)
        ]

    @classmethod
    def move(cls, board, pos):
        turn = cls.get_turn(board) * -1
        board[pos] = turn

    @classmethod
    def random_move(cls, board):
        args = cls.get_args(board)
        if not args:
            return False
        pos = random.choice(args)
        cls.move(board, pos)
        return True

    def __init__(self, epoch=10000) -> None:
        self.epoch = epoch
        self.values = {}

    def new_state(self):
        return None

    def load(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as file:
            # self.values = json.loads(file.read())
            self.values = pickle.loads(file.read())

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            # file.write(json.dumps(self.values, indent=4).encode('ascii'))
            file.write(pickle.dumps(self.values))

    def train(self, board=None):
        if board is None:
            board = self.new_board()

        hash = self.get_hash(board)
        if hash not in self.values:
            self.values[hash] = self.new_state()
        else:
            return

        for pos in self.get_args(board):
            item = board.copy()
            self.move(item, pos)
            if self.is_final(item):
                break
            self.train(item)

    def evaluate(self, board):
        final = self.is_final(board=board)
        if final == BLACK:
            return 1
        elif final == WHITE:
            return -1
        return 0

    def next(self, board):
        turn = self.get_turn(board)
        args = self.get_args(board)
        results = []
        for pos in args:
            item = board.copy()
            self.move(item, pos)
            score = self.evaluate(item)
            results.append((pos, score))

        if not results:
            return None

        if turn == BLACK:
            reverse = False
        else:
            reverse = True

        results = sorted(results, key=lambda e: e[1], reverse=reverse)
        return results[0][0]


class StatisticModel(Model):

    def __init__(self, epoch=10000) -> None:
        super().__init__(epoch)
        self.values = {}

    def new_state(self):
        state = {}
        state['black'] = 0
        state['white'] = 0
        state['draw'] = 0
        return state

    def evaluate(self, board):
        hash = self.get_hash(board)
        if hash in self.values:
            state = self.values[hash]
            return self.probability(state)

        # logger.warning("has not in values %d", hash)
        return super().evaluate(board)

    def probability(self, state):
        count = state['black'] - state['white']
        total = state['black'] + state['white'] + state['draw']
        return count / total

    def train(self, board=None, results=None):

        if board is None:
            board = self.new_board()
        if results is None:
            for _ in tqdm(range(self.epoch)):
                results = {}
                self.train(board, results)
            return

        hash = self.get_hash(board)
        if hash not in self.values:
            self.values[hash] = self.new_state()
        results[hash] = self.values[hash]

        final = self.is_final(board)
        if not final:
            item = board.copy()
            if not self.random_move(item):
                return
            self.train(item, results)
            return

        if final == BLACK:
            key = 'black'
        elif final == WHITE:
            key = 'white'
        elif final == DRAW:
            key = 'draw'

        for hash, state in results.items():
            state[key] += 1


class EnhanceModel(StatisticModel):

    def __init__(self, epoch=10000) -> None:
        super().__init__(epoch)
        self.learning_rate = 0.8

    def random_move(self, board):
        args = self.get_args(board)
        if not args:
            return False

        if (random.uniform(0, 1) > self.learning_rate):
            pos = random.choice(args)
        else:
            pos = self.next(board)

        self.move(board, pos)
        return True


if __name__ == "__main__":

    model = StatisticModel()
    model.train()
    print(len(model.values))
