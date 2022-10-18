from inspect import stack
import os
import sys

from logger import logger
import context
import board
import game

DIRNAME = os.path.dirname(__file__)
MODELPATH = os.path.join(DIRNAME, 'model.pickle')


class GameSignal(board.QtCore.QObject):

    reset = board.QtCore.Signal(None)
    hint = board.QtCore.Signal(None)
    undo = board.QtCore.Signal(None)
    load = board.QtCore.Signal(None)
    save = board.QtCore.Signal(None)
    train = board.QtCore.Signal(None)
    debug = board.QtCore.Signal(None)


class GameContextMenu(context.BaseContextMenu):

    items = [
        ('开局', 'Ctrl+N', lambda self: self.signal.reset.emit(), True),
        ('提示', 'Ctrl+H', lambda self: self.signal.hint.emit(), True),
        ('悔棋', 'Ctrl+Z', lambda self: self.signal.undo.emit(), True),
        ('载入', 'Ctrl+O', lambda self: self.signal.load.emit(), True),
        ('保存', 'Ctrl+S', lambda self: self.signal.save.emit(), True),
        ('训练', 'Ctrl+T', lambda self: self.signal.train.emit(), True),
        'separator',
        ('调试', 'Ctrl+D', lambda self: self.signal.debug.emit(), False),
    ]


class Game(board.BoardFrame, context.ContextMenuMixin):

    menu_class = GameContextMenu
    model_class = game.StatisticModel
    epoch = 10000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui.callback = self.clickPosition

        # self.setWindowOpacity(0.85)
        self.setWindowIcon(board.QtGui.QIcon(board.FAVICON))
        self.setWindowTitle(u"Triplet")
        self.resize(self.ui.size())

        self.reset()

        self.mbox = board.QtWidgets.QMessageBox(self)
        self.signal = GameSignal()
        self.signal.reset.connect(self.reset)
        self.signal.hint.connect(self.hint)
        self.signal.undo.connect(self.undo)
        self.signal.save.connect(self.save)
        self.signal.load.connect(self.load)
        self.signal.train.connect(self.train)
        self.initmenu(self, self.signal)

        self.model = self.model_class(self.epoch)
        self.load()

        self.step = 2

    @board.QtCore.Slot()
    def reset(self):
        self.board = game.Model.new_board()
        self.ui.setBoard(self.board, None)
        self.stack = []

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
        logger.info("train starting...")
        self.model.train(self.board)
        self.save()
        logger.info("train finished value count %d....",
                    len(self.model.values))

    @board.QtCore.Slot()
    def hint(self):
        pos = self.model.next(self.board)
        if not pos:
            return
        self.clickPosition(pos)

    @board.QtCore.Slot()
    def undo(self):
        if not self.stack:
            return

        for _ in range(self.step):
            if not self.stack:
                break
            self.board, pos = self.stack.pop()

        self.ui.setBoard(self.board, pos)

    def check(self):
        if not game.Model.get_args(self.board):
            return
        final = game.Model.is_final(self.board)
        if final == game.BLACK:
            self.mbox.about(self, "信息", "黑胜")
            return True
        elif final == game.WHITE:
            self.mbox.about(self, "信息", "白胜")
            return True
        return False

    def move(self, pos):
        self.stack.append((self.board.copy(), pos))
        game.Model.move(self.board, pos)
        self.ui.setBoard(self.board, pos)
        if self.check():
            return True
        return False

    def clickPosition(self, pos):
        if self.check():
            return
        if self.board[pos]:
            return

        if self.move(pos):
            return

        if self.step == 1:
            return

        pos = self.model.next(self.board)
        if not pos:
            return
        if self.move(pos):
            return


def start_app():
    app = board.QtWidgets.QApplication(sys.argv)
    ui = Game()
    ui.show()
    app.exec_()


def main():
    start_app()


if __name__ == "__main__":
    main()
