
# import torch
# from torch import nn
# from tqdm import tqdm


# DIRNAME = os.path.dirname(__file__)
# STATE_PATH = os.path.join(DIRNAME, "model.pt")


# if torch.cuda.is_available():
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
#     print("using cuda:", torch.cuda.get_device_name(0))
# else:
#     torch.set_default_tensor_type(torch.FloatTensor)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info("cuda available %s device %s", torch.cuda.is_available(), device)


# class TripletModule(nn.Module):

#     def __init__(self) -> None:
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(9, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 1)
#         )

#         self.loss = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.5)

#     def forward(self, inputs):
#         y = self.model(inputs)
#         return y

#     def train(self, inputs, labels):
#         outputs = self.forward(inputs)
#         loss = self.loss(outputs, labels)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return loss

#     def evaluate(self, item, turn):
#         res = item.flatten()
#         return float(self.forward(torch.from_numpy(res).float())[0])


# class TripletDataSet(torch.utils.data.Dataset):

#     def __init__(self, length=10000) -> None:
#         super().__init__()
#         self.counter = 0
#         self.items = []
#         self.length = length

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         if index > len(self):
#             raise StopIteration

#         if not self.items:
#             self.generate()
#             idx = index % len(self.items)
#         else:
#             idx = index % len(self.items)
#             if not idx:
#                 self.generate()

#         result = self.items[idx]
#         self.counter += 1
#         return result

#     def move(self, item, turn):
#         args = np.argwhere(item == 0)
#         pos = random.choice(args)
#         item[(pos[0], pos[1])] = turn

#     def evaluate(self, item, turn):
#         results = []

#         for line in item:
#             results.append(sum(line))

#         results.append(item.trace())

#         T = item.transpose()

#         for line in T:
#             results.append(sum(line))
#         results.append(T.trace())

#         if turn > 0:
#             return max(results)
#         else:
#             return min(results)

#     def generate(self):
#         item = np.zeros((3, 3), dtype=np.int8)
#         turn = 1

#         result = 0

#         for _ in range(9):
#             self.move(item, turn)
#             result = self.evaluate(item, turn)
#             turn *= -1
#             self.items.append([item, result])
#             if abs(result) == 3:
#                 break
#             item = item.copy()

#         for item in self.items:
#             item[1] = result / 3

#         for _ in range(10):
#             self.items.append(self.items[-1])

# @board.QtCore.Slot()
# def save(self):
#     logger.info("save model parameters...")
#     torch.save(self.model.state_dict(), STATE_PATH)

# @board.QtCore.Slot()
# def load(self):
#     logger.info("load model parameters...")
#     if os.path.exists(STATE_PATH):
#         self.model.load_state_dict(torch.load(STATE_PATH))

# def train(self):
#     with tqdm(self.dataset) as bar:
#         for inputs, label in bar:
#             loss = self.model.train(
#                 torch.from_numpy(inputs.flatten()).float(),
#                 torch.FloatTensor([label, ]))
#             bar.set_postfix(loss=loss.item())
