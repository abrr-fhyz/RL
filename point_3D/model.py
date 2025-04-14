import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from vpython import sphere, vector, rate, scene, arrow, color, mag

class DQN(nn.Module):
    def __init__(self, stateSize, actionSize, hiddenSize = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(stateSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, actionSize)
        )

    def forward(self, state):
        return self.net(state)

class RLModel:
    def __init__(self, stateSize, actionSize, learningRate = 1e-3, gamma = 0.99, bufferSize = 10000, batchSize = 64):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.gamma = gamma
        self.batchSize = batchSize
        self.memory = deque(maxlen = bufferSize)

        self.device = torch.device("cpu")
        self.policyNet = DQN(stateSize, actionSize).to(self.device)
        self.targetNet = DQN(stateSize, actionSize).to(self.device)
        self.targetNet.load_state_dict(self.policyNet.state_dict())
        self.targetNet.eval()

        self.optimizer = optim.Adam(self.policyNet.parameters(), lr = learningRate)
        self.lossFunction = nn.SmoothL1Loss()

    def remember(self, state, action, reward, nextstate, done):
        self.memory.append((state, action, reward, nextstate, done))

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.randrange(self.actionSize)
        stateTensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qValues = self.policyNet(stateTensor)
        return int(torch.argmax(qValues).item())

    def replay(self):
        if len(self.memory) < self.batchSize:
            return

        batch = random.sample(self.memory, self.batchSize)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        nextStates = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch]).astype(np.float32)

        stateBatch = torch.from_numpy(states).float().to(self.device)
        actionBatch = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewardBatch = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        nextStateBatch = torch.from_numpy(nextStates).float().to(self.device)
        doneBatch = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        qValues = self.policyNet(stateBatch).gather(1, actionBatch)
        with torch.no_grad():
            nextQValues = self.targetNet(nextStateBatch).max(1)[0].unsqueeze(1)
            target = rewardBatch + (1 - doneBatch) * self.gamma * nextQValues

        loss = self.lossFunction(qValues, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def updateTargetNetwork(self):
        self.targetNet.load_state_dict(self.policyNet.state_dict())

