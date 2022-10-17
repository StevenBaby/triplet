from asyncio import selector_events
import os
import sys
import random
# import pickle
import json

import numpy as np
from tqdm import tqdm
from zmq import has
from logger import logger
import context
import board

DIRNAME = os.path.dirname(__file__)
MODELPATH = os.path.join(DIRNAME, 'model.json')


class Model(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def turn(item):
        black = len(np.argwhere(item == 1))
        white = len(np.argwhere(item == -1))
        if (black - white) not in (0, 1):
            raise Exception("invalid item %s", item)

        if (black - white) == 1:
            turn = 1
        else:
            turn = -1
        return turn

    @staticmethod
    def evaluate(item):
        results = []

        turn = Model.turn(item=item)

        for line in item:
            results.append(sum(line))

        results.append(sum([
            item[(0, 0)],
            item[(1, 1)],
            item[(2, 2)],
        ]))

        results.append(sum([
            item[(2, 0)],
            item[(1, 1)],
            item[(0, 2)],
        ]))

        T = item.transpose()

        for line in T:
            results.append(sum(line))
        count = len(np.argwhere(item == 0))

        if turn > 0:
            result = max(results)
        else:
            result = min(results)

        if not count and abs(result) != 3:
            result = 0

        return result / 3.0


class GameModel(Model):

    def __init__(self) -> None:
        super().__init__()
        self.learning_rate = 0.5
        self.epochs = 1000000
        self.values = {}

    @staticmethod
    def item_hash(item):
        result = 0
        for line in item:
            for var in line:
                result <<= 2
                result |= (int(var) + 2)
        return result

    # 利用
    def exploit(self, item):
        args = np.argwhere(item == 0)
        turn = self.turn(item=item) * -1

        results = []
        for pos in args:
            var = item.copy()
            idx = (pos[0], pos[1])
            var[idx] = turn
            score = self.evaluate(var)
            results.append((idx, score))

        if not results:
            return None

        if turn > 0:
            reverse = True
        else:
            reverse = False

        results = sorted(results, key=lambda e: e[1], reverse=reverse)
        return results[0][0]

    # 探索
    def explore(self, item):
        args = np.argwhere(item == 0)
        pos = random.choice(args)
        return (pos[0], pos[1])

    def evaluate(self, item):
        idx = self.item_hash(item)
        if idx not in self.values:
            self.values[idx] = Model.evaluate(item)
            logger.debug("params count %s", len(self.values))
        return self.values[idx]

    def load(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as file:
            self.values = json.loads(file.read())

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            file.write(json.dumps(self.values, indent=4).encode('ascii'))

    def train(self):
        for _ in tqdm(range(self.epochs)):
            item = np.zeros((3, 3), dtype=np.int8)
            count = 9
            while count:
                idx = self.item_hash(item)
                if abs(self.evaluate(item=item)) == 1.0:
                    break
                if not count:
                    self.values[idx] = 0
                    break

                turn = self.turn(item) * -1

                if self.learning_rate < random.uniform(0, 1):
                    pos = self.exploit(item)
                    item[pos] = turn
                else:
                    pos = self.explore(item)
                    item[pos] = turn
                value = self.evaluate(item=item)
                if abs(value) == 1.0:
                    # self.values[idx] = value
                    # else:
                    offset = 1 - self.values[idx]
                    self.values[idx] = self.learning_rate * offset

                count = len(np.argwhere(item == 0))


class ExModel(GameModel):

    def evaluate(self, item):
        idx = self.item_hash(item)
        assert (idx in self.values)
        return self.values[idx]

    def minmax(self, item):
        idx = self.item_hash(item)
        if idx in self.values:
            return self.values[idx]

        value = Model.evaluate(item)
        if abs(value) == 1.0:
            self.values[idx] = value
            return value

        args = np.argwhere(item == 0)
        if len(args) == 0:
            self.values[idx] = 0
            return 0

        turn = self.turn(item) * -1
        best = turn * -1

        for var in args:
            pos = (var[0], var[1])
            step = item.copy()
            step[pos] = turn
            value = self.minmax(step)

            if turn > 0 and value > best:
                best = value

            if turn < 0 and value < best:
                best = value
        self.values[idx] = best
        return best

    def train(self):
        item = np.zeros((3, 3), dtype=np.int8)
        self.minmax(item)
        logger.info("training finished!!!")


class GameSignal(board.QtCore.QObject):

    reset = board.QtCore.Signal(None)
    load = board.QtCore.Signal(None)
    save = board.QtCore.Signal(None)
    train = board.QtCore.Signal(None)
    debug = board.QtCore.Signal(None)


class GameContextMenu(context.BaseContextMenu):

    items = [
        ('开局', 'Ctrl+N', lambda self: self.signal.reset.emit(), True),
        ('载入', 'Ctrl+O', lambda self: self.signal.load.emit(), True),
        ('保存', 'Ctrl+S', lambda self: self.signal.save.emit(), True),
        ('训练', 'Ctrl+T', lambda self: self.signal.train.emit(), True),
        'separator',
        ('调试', 'Ctrl+D', lambda self: self.signal.debug.emit(), False),
    ]


class Game(board.QtWidgets.QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = board.Board(self, callback=self.clickPosition)
        # self.setWindowOpacity(0.85)
        self.setWindowIcon(board.QtGui.QIcon(board.FAVICON))
        self.setWindowTitle(u"Triplet")
        self.resize(self.ui.size())

        self.reset()
        # self.model = GameModel()
        self.model = ExModel()

        self.mbox = board.QtWidgets.QMessageBox(self)
        self.signal = GameSignal()
        self.menu = GameContextMenu(self, self.signal)
        self.setupContextMenu()

        self.signal.reset.connect(self.reset)
        self.signal.save.connect(self.save)
        self.signal.load.connect(self.load)
        self.signal.train.connect(self.train)

    def setupContextMenu(self):
        self.setContextMenuPolicy(board.QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, point):
        self.menu.exec_(self.mapToGlobal(point))

    @board.QtCore.Slot()
    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.turn = -1
        self.result = 0
        self.ui.setBoard(self.board, None)

    @board.QtCore.Slot()
    def save(self):
        logger.info("save model parameters...")
        self.model.save(MODELPATH)

    @board.QtCore.Slot()
    def load(self):
        logger.info("load model parameters...")
        self.model.load(MODELPATH)

    @board.QtCore.Slot()
    def train(self):
        logger.info("training...")
        self.model.train()

    def move(self, pos):
        if self.board[pos] != 0:
            return False

        self.turn *= -1
        self.board[pos] = self.turn

    def next_move(self):
        args = np.argwhere(self.board == 0)
        turn = self.turn * -1

        results = []
        for pos in args:
            item = self.board.copy()
            idx = (pos[0], pos[1])
            item[idx] = turn
            score = self.model.evaluate(item)
            results.append((idx, score))

        if not results:
            return None

        if turn > 0:
            reverse = True
        else:
            reverse = False

        results = sorted(results, key=lambda e: e[1], reverse=reverse)
        return results[-1][0]

    def check(self):
        if not len(np.argwhere(self.board == 0)):
            return
        if abs(self.result) != 1.0:
            self.result = Model.evaluate(self.board)
        if self.result == 1.0:
            self.mbox.about(self, "信息", "黑胜")
            return True
        elif self.result == -1.0:
            self.mbox.about(self, "信息", "白胜")
            return True
        return False

    def clickPosition(self, pos):
        if self.check():
            return
        if self.board[pos]:
            return

        self.move(pos)
        self.ui.setBoard(self.board, pos)
        if self.check():
            return

        pos = self.next_move()
        if not pos:
            return
        self.move(pos)
        self.ui.setBoard(self.board, pos)
        if self.check():
            return

    def resizeEvent(self, event):
        self.ui.resizeEvent(event)
        return super().resizeEvent(event)


def start_app():
    app = board.QtWidgets.QApplication(sys.argv)
    ui = Game()
    ui.show()
    app.exec_()


def main():
    start_app()
    model = ExModel()
    # model.train()


if __name__ == "__main__":
    main()
