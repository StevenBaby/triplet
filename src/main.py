import os
import sys
import random

import numpy as np

import torch
from torch import nn
from tqdm import tqdm


from logger import logger
import context
import board

DIRNAME = os.path.dirname(__file__)
STATE_PATH = os.path.join(DIRNAME, "model.pt")


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("using cuda:", torch.cuda.get_device_name(0))
else:
    torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("cuda available %s device %s", torch.cuda.is_available(), device)


class TripletModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.5)

    def forward(self, inputs):
        y = self.model(inputs)
        return y

    def train(self, inputs, labels):
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, item, turn):
        res = item.flatten()
        return float(self.forward(torch.from_numpy(res).float())[0])


class TripletDataSet(torch.utils.data.Dataset):

    def __init__(self, length=10000) -> None:
        super().__init__()
        self.counter = 0
        self.items = []
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > len(self):
            raise StopIteration

        if not self.items:
            self.generate()
            idx = index % len(self.items)
        else:
            idx = index % len(self.items)
            if not idx:
                self.generate()

        result = self.items[idx]
        self.counter += 1
        return result

    def move(self, item, turn):
        args = np.argwhere(item == 0)
        pos = random.choice(args)
        item[(pos[0], pos[1])] = turn

    def evaluate(self, item, turn):
        results = []

        for line in item:
            results.append(sum(line))

        results.append(item.trace())

        T = item.transpose()

        for line in T:
            results.append(sum(line))
        results.append(T.trace())

        if turn > 0:
            return max(results)
        else:
            return min(results)

    def generate(self):
        item = np.zeros((3, 3), dtype=np.int8)
        turn = 1

        result = 0

        for _ in range(9):
            self.move(item, turn)
            result = self.evaluate(item, turn)
            turn *= -1
            self.items.append([item, result])
            if abs(result) == 3:
                break
            item = item.copy()

        for item in self.items:
            item[1] = result / 3

        for _ in range(10):
            self.items.append(self.items[-1])


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
        self.model = TripletModule()
        self.dataset = TripletDataSet(100000)
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
        torch.save(self.model.state_dict(), STATE_PATH)

    @board.QtCore.Slot()
    def load(self):
        logger.info("load model parameters...")
        if os.path.exists(STATE_PATH):
            self.model.load_state_dict(torch.load(STATE_PATH))

    def train(self):
        with tqdm(self.dataset) as bar:
            for inputs, label in bar:
                loss = self.model.train(
                    torch.from_numpy(inputs.flatten()).float(),
                    torch.FloatTensor([label, ]))
                bar.set_postfix(loss=loss.item())

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
            score = self.model.evaluate(item, turn)
            results.append((idx, score))

        if not results:
            return None

        if turn > 0:
            reverse = True
        else:
            reverse = False

        results = sorted(results, key=lambda e: e[1], reverse=reverse)
        return results[0][0]

    def clickPosition(self, pos):
        if self.board[pos]:
            return

        self.move(pos)
        self.ui.setBoard(self.board, pos)
        pos = self.next_move()
        if not pos:
            return
        self.move(pos)
        self.ui.setBoard(self.board, pos)

    def resizeEvent(self, event):
        self.ui.resizeEvent(event)
        return super().resizeEvent(event)


def main():
    app = board.QtWidgets.QApplication(sys.argv)
    ui = Game()
    ui.show()
    app.exec_()


if __name__ == "__main__":
    main()
