import os
import sys
import random

import numpy as np
from logger import logger
import context
import board
import toast


class Model(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def evaluate(item):
        results = []

        black = len(np.argwhere(item == 1))
        white = len(np.argwhere(item == -1))
        if (black - white) not in (0, 1):
            raise Exception("invalid item %s", item)

        if (black - white) == 1:
            turn = 1
        else:
            turn = -1

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

        if turn > 0:
            return max(results) / 3.0
        else:
            return min(results) / 3.0


class GameSignal(board.QtCore.QObject):

    load = board.QtCore.Signal(None)
    save = board.QtCore.Signal(None)
    train = board.QtCore.Signal(None)
    debug = board.QtCore.Signal(None)


class GameContextMenu(context.BaseContextMenu):

    items = [
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

        self.board = np.zeros((3, 3), dtype=np.int8)
        self.turn = -1
        self.result = 0
        self.model = Model()

        self.toast = toast.Toast(self)
        self.signal = GameSignal()
        self.menu = GameContextMenu(self, self.signal)
        self.setupContextMenu()

        self.signal.save.connect(self.save)
        self.signal.load.connect(self.load)
        self.signal.train.connect(self.train)

    def setupContextMenu(self):
        self.setContextMenuPolicy(board.QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, point):
        self.menu.exec_(self.mapToGlobal(point))

    @board.QtCore.Slot()
    def save(self):
        logger.info("save model parameters...")
        self.model.save()

    @board.QtCore.Slot()
    def load(self):
        logger.info("load model parameters...")
        self.model.load()

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
        return results[0][0]

    def check(self):
        if abs(self.result) != 1.0:
            self.result = Model.evaluate(self.board)
            logger.debug("evaluate result %s", self.result)
        if self.result == 1.0:
            self.toast.message("黑胜")
            return True
        elif self.result == -1.0:
            self.toast.message("白胜")
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


if __name__ == "__main__":
    main()
