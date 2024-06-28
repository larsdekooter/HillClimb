import torch
import torch.nn as nn
import os
from collections import deque
import numpy as np
import random
import data
import torch.optim as optim


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
    def __init__(self, model) -> None:
        self.lr = data.lr
        self.gamma = data.gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.model = model

    def trainStep(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state.shape) == 1:  # if one state gets passed in
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class Network:
    def __init__(self) -> None:
        self.gamma = data.gamma
        self.memory = deque(maxlen=100_000)
        self.model = LinearQNet(2, data.hiddenSize, data.hiddenSize, 3)
        self.trainer = QTrainer(self.model)
        self.net = 0
        self.rand = 0
        self.decayStep = 0

    def getMove(self, observation):
        epsilon = data.minEpsilon + (data.maxEpsilon - data.minEpsilon) * np.exp(
            -data.decayRate * self.decayStep
        )

        if np.random.rand() < epsilon:
            move = random.randint(0, 2)
            self.rand += 1
        else:
            with torch.no_grad():
                state0 = torch.tensor(observation, dtype=torch.float)
                move = self.model(state0).argmax().item()
                self.net += 1

        self.decayStep += 1
        return move

    def trainShort(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)

    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def trainLong(self):
        if len(self.memory) < data.batchSize:
            return

        batch = random.sample(self.memory, data.batchSize)
        states, actions, rewards, nextStates, dones = zip(*batch)
        self.trainer.trainStep(
            np.array(states), actions, rewards, np.array(nextStates), dones
        )
