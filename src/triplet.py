
import re


import random
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt
from board import BoardFrame


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("using cuda:", torch.cuda.get_device_name(0))
else:
    torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TripletModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 512),
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


class TripletDataSet(torch.utils.data.Dataset):

    def __init__(self, length) -> None:
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

    def move(self, item, turn):
        args = np.argwhere(item == 0)
        pos = random.choice(args)
        item[(pos[0], pos[1])] = turn

    def generate(self):
        item = np.zeros((3, 3), dtype=np.int8)
        turn = 1

        result = 0

        for _ in range(9):
            self.move(item, turn)
            result = self.evaluate(item, turn)
            res = np.hstack((item.flatten(), np.array([turn])))
            turn *= -1
            self.items.append([res, result])
            if abs(result) == 3:
                break
            item = item.copy()

        for item in self.items:
            item[1] = result / 3

        for _ in range(10):
            self.items.append(self.items[-1])


class Triplet(object):

    def __init__(self, board=None):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.turn = -1
        self.model = TripletModule()
        self.dataset = TripletDataSet(10000)

    def move(self, pos):
        if self.board[pos] != 0:
            return False

        self.turn *= -1
        self.board[pos] = self.turn

    def input_move(self):
        line = input("please input pos: ")
        res = re.search(r"([012]).*([012])", line)
        if res:
            return (int(res.group(1)), int(res.group(2)))

    def next(self):
        args = np.argwhere(self.board == 0)
        turn = self.turn * -1

        results = []
        for pos in args:
            item = self.board.copy()
            idx = (pos[0], pos[1])
            item[idx] = turn
            score = evaluate(item, turn)
            results.append((idx, score))

        if turn > 0:
            reverse = True
        else:
            reverse = False

        results = sorted(results, key=lambda e: e[1], reverse=reverse)
        return results[0][0]
