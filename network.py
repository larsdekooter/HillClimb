import torch
import torch.nn as nn
import os
from collections import deque, namedtuple
import numpy as np
import random
import data
import torch.optim as optim


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "nextState", "done")
)


class Memory(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)

    def last(self):
        return self.memory[-1]


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
        )

    def forward(self, x):
        return self.stack(x)

    def save(self, filename="model.pth"):
        modelFolderPath = "./model"
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        filename = os.path.join(modelFolderPath, filename)
        torch.save(self.state_dict(), filename)


class QTrainer:
    def __init__(self, policyModel, targetModel) -> None:
        self.lr = data.lr
        self.gamma = data.gamma
        self.optimizer = optim.AdamW(policyModel.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.policyModel = policyModel
        self.targetModel = targetModel

    def updateModels(self):
        tDict = self.targetModel.state_dict()
        pDict = self.policyModel.state_dict()

        for key in pDict:
            tDict[key] = pDict[key] * data.tau + tDict[key] * (1 - data.tau)
        self.targetModel.load_state_dict(tDict)

    def torchTrainStep(self, transitions):
        batch = Transition(*zip(*transitions))

        nonFinalMask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.nextState)), dtype=torch.bool
        )
        nonFinalNextStates = torch.cat([s for s in batch.nextState if s is not None])

        stateBatch = torch.cat(batch.state)
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)
        stateActionValues = self.policyModel(stateBatch).gather(1, actionBatch)
        nextStateValues = torch.zeros(data.batchSize)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = (
                self.targetModel(nonFinalNextStates).max(1).values
            )

        expectedStateActionValues = (nextStateValues * data.gamma) + rewardBatch
        loss = self.criterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policyModel.parameters(), 100)
        self.optimizer.step()

    def torchTrainStep1(self, batch):

        nonFinalMask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.nextState)), dtype=torch.bool
        )
        nonFinalNextStates = torch.cat([s for s in batch.nextState if s is not None])

        stateBatch = torch.cat(batch.state)
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)
        stateActionValues = self.policyModel(stateBatch).gather(1, actionBatch)
        nextStateValues = torch.zeros(1)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = (
                self.targetModel(nonFinalNextStates).max(1).values
            )

        expectedStateActionValues = (nextStateValues * data.gamma) + rewardBatch
        loss = self.criterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policyModel.parameters(), 100)
        self.optimizer.step()


class Network:
    def __init__(self) -> None:
        device = torch.device("cpu")
        self.gamma = data.gamma
        self.memory = Memory(10000)
        self.policyModel = LinearQNet(2, data.hiddenSize, data.hiddenSize, 3).to(device)
        if os.path.exists("./model/model.pth"):
            self.policyModel.load_state_dict(torch.load("./model/model.pth"))
        self.targetModel = LinearQNet(2, data.hiddenSize, data.hiddenSize, 3).to(device)
        self.targetModel.load_state_dict(self.policyModel.state_dict())
        self.trainer = QTrainer(self.policyModel, self.targetModel)
        self.net = 0
        self.rand = 0
        self.decayStep = 0

    def getMove(self, observation):
        epsilon = data.minEpsilon + (data.maxEpsilon - data.minEpsilon) * np.exp(
            -data.decayRate * self.decayStep
        )

        if np.random.random() < epsilon:
            move = random.randint(0, 2)
            self.rand += 1
        else:
            with torch.no_grad():
                state0 = torch.tensor(observation, dtype=torch.float)
                move = self.policyModel(state0).argmax().item()
                self.net += 1

        self.decayStep += 1
        return move

    def trainLong(self):
        if len(self.memory) < data.batchSize:
            return
        transitions = self.memory.sample(data.batchSize)
        self.trainer.torchTrainStep(transitions)

    def trainShort(self):
        transitions = [self.memory.last()]
        batch = Transition(*zip(*transitions))
        if batch.nextState[0] == None:
            return
        self.trainer.torchTrainStep(batch)

    def remember(self, state, action, reward, nextState, done):
        self.memory.push(
            torch.tensor(np.array([state]), dtype=torch.float32),
            torch.tensor(np.array([action]), dtype=torch.int64).unsqueeze(1),
            torch.tensor(np.array([reward]), dtype=torch.float32),
            (
                torch.tensor(np.array([nextState]), dtype=torch.float32)
                if nextState is not None
                else None
            ),
            torch.tensor(np.array([done]), dtype=torch.int),
        )

    # def trainLong(self):
    #     if len(self.memory) < data.batchSize:
    #         return

    #     batch = random.sample(self.memory, data.batchSize)
    #     states, actions, rewards, nextStates, dones = zip(*batch)
    #     self.trainer.trainSteps(states, actions, rewards, nextStates, dones)
