# coding=utf-8
'''
(C) Copyright 2021 Steven;
@author: Steven kangweibaby@163.com
@date: 2021-05-31

PySide2 棋盘基础控件，只用于棋盘的展示，和点击回调。
'''

import sys
import os
import ctypes

from PySide2 import QtCore, QtWidgets, QtGui
from PySide2.QtWidgets import QLabel
from PySide2.QtGui import QPixmap
import numpy as np

from logger import logger


DIRNAME = os.path.dirname(__file__)
IMAGEDIR = os.path.abspath(os.path.join(DIRNAME, "../images"))

BOARD_IMAGE = os.path.join(IMAGEDIR, 'board.png')
BLACK_IMAGE = os.path.join(IMAGEDIR, "black.png")
BLACK_IMAGE_BORDER = os.path.join(IMAGEDIR, "black_border.png")
WHITE_IMAGE = os.path.join(IMAGEDIR, "white.png")
WHITE_IMAGE_BORDER = os.path.join(IMAGEDIR, "white_border.png")
FAVICON = os.path.join(IMAGEDIR, "favicon.ico")

SPAN = 3


class Board(QLabel):

    '''
    棋盘坐标与屏幕坐标类似，左上角为 (0, 0)，右下角为 (8, 9)
    '''

    flags = QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint

    def __init__(self, parent=None, callback=None):
        super().__init__(parent=parent)

        self.csize = 60
        self.setPixmap(QtGui.QPixmap(BOARD_IMAGE))

        self.chesses = {
            1: QtGui.QPixmap(BLACK_IMAGE),
            -1: QtGui.QPixmap(WHITE_IMAGE),
        }

        self.border_chesses = {
            1: QtGui.QPixmap(BLACK_IMAGE_BORDER),
            -1: QtGui.QPixmap(WHITE_IMAGE_BORDER),
        }

        if parent is None:
            self.setWindowIcon(QtGui.QIcon(FAVICON))
            self.setWindowTitle(u"Triplet")
            self.setWindowFlags(self.flags)

        # https://www.mfitzp.com/tutorials/packaging-pyqt5-pyside2-applications-windows-pyinstaller/

        app = QtWidgets.QApplication.instance()
        if app:
            app.setWindowIcon(QtGui.QIcon(FAVICON))

        self.setObjectName(u"Board")
        self.setScaledContents(True)

        self.resize(self.csize * SPAN, self.csize * SPAN)

        self.board = np.zeros((SPAN, SPAN), dtype=np.int8)
        self.where = (1, 1)
        self.labels = np.zeros((SPAN, SPAN), dtype=QLabel)

        self.callback = callback
        self.signal = QtCore.Signal()

        self.refresh()

    def setBoard(self, board, where=None):
        # 设置 棋盘 board，以及该步的位置 pos
        # 由于 该函数可能在多个线程中调用，所以下面触发 signal.refresh
        # QT 会自动将刷新棋盘的重任放到主线程去做
        # 如果直接在非主线程调用 refresh 函数，程序可能莫名其妙的死掉。

        self.board = board
        self.where = where
        self.signal.emit()

    @QtCore.Slot()
    def refresh(self):
        # 刷新棋盘
        for pos in np.argwhere(self.board != 0):
            self.setChess((pos[0], pos[1]))

        super().update()

    def resizeEvent(self, event):
        # 窗口大小变化之后，修改棋盘和棋子的大小

        w = self.parentWidget().width()
        h = self.parentWidget().height()

        height = h
        width = h

        if width > w:
            width = w
            height = w

        x = (w - width) // 2
        y = (h - height) // 2
        self.setGeometry(x, y, width, height)

        self.csize = width // SPAN

        self.refresh()

    def mousePressEvent(self, event):
        # 鼠标点击事件
        # 只处理鼠标左键点击

        if event.buttons() != QtCore.Qt.LeftButton:
            return super().mousePressEvent(event)

        # 获取点击的棋盘坐标
        pos = self.getPosition(event)
        if not pos:
            return
        logger.debug("click %s", pos)
        self.clickPosition(pos)

    def clickPosition(self, pos):
        if callable(self.callback):
            self.callback(pos)

    def setChess(self, pos):
        # 将某个位置设置成某个棋子

        label = self.labels[pos]
        if not label:
            label = QLabel(self)
            label.pos = pos
            self.labels[pos] = label

        chess = self.board[pos]
        if not chess:
            label.setVisible(False)
            return

        if self.pos == pos:
            images = self.border_chesses
        else:
            images = self.chesses
        image = images[chess]
        label.setPixmap(image)
        label.setScaledContents(True)
        label.setGeometry(self.getChessGeometry(pos))
        label.setVisible(True)

    def getChessGeometry(self, pos):
        # 获取某个位置棋子在棋盘的坐标及尺寸

        return QtCore.QRect(
            pos[0] * self.csize,
            pos[1] * self.csize,
            self.csize,
            self.csize
        )

    def getPosition(self, event):
        # 通过鼠标位置，获取棋子坐标的位置

        x = event.x() // self.csize
        y = event.y() // self.csize

        if x < 0 or x >= SPAN:
            return None
        if y < 0 or y >= SPAN:
            return None

        pos = (int(x), int(y))
        return pos


class BoardFrame(QtWidgets.QFrame):

    def __init__(self, parent=None, board_class=Board):
        super().__init__(parent)
        self.board = board_class(self)
        # self.setWindowOpacity(0.85)
        if parent is None:
            self.setWindowIcon(QtGui.QIcon(FAVICON))
            self.setWindowTitle(u"Triplet")
        self.resize(self.board.size())

    def resizeEvent(self, event):
        self.board.resizeEvent(event)
        return super().resizeEvent(event)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = BoardFrame()
    ui.show()
    app.exec_()


if __name__ == '__main__':
    main()
