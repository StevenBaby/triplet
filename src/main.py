import os
import sys

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from logger import logger

from triplet import device
from triplet import TripletModule
from triplet import TripletDataSet


DIRNAME = os.path.dirname(__file__)
STATE_PATH = os.path.join(DIRNAME, "model.pt")


def main():
    logger.info("triplet start")
    model = TripletModule()
    dataset = TripletDataSet(100)

    with tqdm(dataset) as bar:
        for inputs, label in bar:
            loss = model.train(torch.from_numpy(inputs).float(),
                               torch.tensor([label]))
            bar.set_postfix(loss=loss.item())


if __name__ == "__main__":
    main()
