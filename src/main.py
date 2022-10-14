# coding=utf8

import re
import torch
from torch import nn
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from logger import logger

values = {}


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
            item[1] = result

    def draw(self, item):
        # plt.imshow(item)
        for line in item:
            for var in line:
                if var == -1:
                    print("🔵", end="")
                elif var == 1:
                    print("🔴", end="")
                else:
                    print("⬜", end="")
            print()


class Triplet(object):

    def __init__(self, board=None):
        self.model = TripletModule()
        self.ds = TripletDataSet()
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.turn = -1

    def train(self):
        dataset = TripletDataSet(length=10000)
        with tqdm(dataset) as bar:
            for inputs, label in bar:
                loss = self.model.train(torch.from_numpy(
                    inputs).float(), torch.FloatTensor([label, ]))
                bar.set_postfix(loss=loss.item())

    def move(self, pos):
        if self.board[pos] != 0:
            return False

        self.turn *= -1
        self.board[pos] = self.turn

    def draw(self):
        self.ds.draw(self.board)

    def evaluate(self, item, turn):
        res = np.hstack((item.flatten(), np.array([turn])))
        return float(self.model.forward(torch.from_numpy(res).float())[0])

    def input_move(self):
        while True:
            line = input("please input pos: ")
            res = re.search(r"([012]).*([012])", line)
            if res:
                return (int(res.group(1)), int(res.group(2)))
            logger.warning("input %s not available!!!", line)


def main():
    logger.info("triplet start")
    triplet = Triplet()
    triplet.train()
    while True:
        triplet.draw()
        pos = triplet.input_move()
        triplet.move(pos)
        print(triplet.evaluate(triplet.board, triplet.turn))


if __name__ == "__main__":
    main()
