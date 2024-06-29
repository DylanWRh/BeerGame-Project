import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import random
from collections import deque 

class QNet(nn.Module):
    def __init__(self, config):
        super(QNet, self).__init__()
        self.fc_layers = nn.ModuleList()
        for i in range(config.NoHiLayer):
            self.fc_layers.append(
                nn.Linear(config.nodes[i], config.nodes[i + 1])
            )
        self.out = nn.Linear(config.nodes[-2], config.nodes[-1])
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x


class DQNAgent:
    def __init__(self, agentNum, config):
        super(DQNAgent, self).__init__()
        self.agentNum = agentNum
        self.config = config

        self.target_QNet = QNet(config) 
        self.QNet = QNet(config)   

        # Optimizer
        self.optimizer = optim.Adam(self.QNet.parameters(), lr=config.lr0)

        # Replay Memory Pool
        self.replayMemory = deque()
        self.replaySize = 0

        # eps-greedy strategy
        if self.config.maxEpisodesTrain != 0:
            self.epsilon = config.epsilonBeg
        else:
            self.epsilon = 0
        self.epsilonRed	 = self.epsilonBuild()

        # Other initialization
        self.currentState = []
        self.timeStep = 0

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, self.config.batchSize)
        state_batch = torch.tensor([data[0] for data in minibatch], dtype=torch.float32)
        action_batch = torch.tensor([data[1] for data in minibatch], dtype=torch.float32)
        reward_batch = torch.tensor([data[2] for data in minibatch], dtype=torch.float32)
        nextState_batch = torch.tensor([data[3] for data in minibatch], dtype=torch.float32)

        # Step 2: calculate y
        QValue_batch = self.target_QNet.forward(nextState_batch).detach().numpy()
        y_batch = reward_batch + (1 - np.array([data[4] for data in minibatch]).astype(np.float32)) * \
              self.config.gamma * np.max(QValue_batch, axis=1)

        # Step 3: Optimize the net
        self.optimizer.zero_grad()
        Q_values = self.QNet.forward(state_batch)
        Q_Action = torch.sum(Q_values * action_batch, dim=1)
        loss = F.mse_loss(Q_Action, y_batch)
        loss.backward()
        self.optimizer.step()

        # Step 4: Save the network
        os.makedirs(self.config.DQNckpt, exist_ok=True)
        if (self.timeStep + 1) % self.config.saveInterval == 0:
            torch.save(self.QNet.state_dict(), os.path.join(self.config.DQNckpt, f'model-{self.timeStep}.pt'))
            print("Network weights are saved")
        if self.timeStep % self.config.dnnUpCnt == 0:
            self.copyTargetQNetwork()
    
    def train(self, nextObservation, action, reward, terminal, playType):
        newState = np.append(self.currentState[1:,:], [nextObservation], axis=0)  

        if playType == "train":
            # Get replay memory
            if self.config.MultiAgent:
                if self.config.MultiAgentRun[self.agentNum]:
                    self.replayMemory.append([self.currentState, action, reward, newState, terminal])
                    self.replaySize = len(self.replayMemory)
            else:
                self.replayMemory.append([self.currentState, action, reward, newState, terminal])
                self.replaySize = len(self.replayMemory)

            # Maintrain the replay pool
            if self.replaySize > self.config.maxReplayMem and self.config.MultiAgentRun[self.agentNum]:
                self.replayMemory.popleft()
                self.trainQNetwork()
                state = "train"
                self.timeStep += 1
            elif self.replaySize >= self.config.minReplayMem and self.config.MultiAgentRun[self.agentNum]:
                state = "train"
                self.trainQNetwork()
                self.timeStep += 1
            else:
                state = "observe" 
                
            if terminal and state == "train":
                self.epsilonReduce()

        self.currentState = newState

    def getDNNAction(self, playType):
        action = np.zeros(self.config.actionListLen)
        action_index = 0

        if playType == "train":
            # eps-greedy algorithm
            if (random.random() <= self.epsilon) or (self.replaySize < self.config.minReplayMem):
                action_index = random.randrange(self.config.actionListLen)
                action[action_index] = 1
            else:
                Q_values = self.QNet.forward(torch.tensor([self.currentState], dtype=torch.float32)).detach().numpy()[0]
                action_index = np.argmax(Q_values)
                action[action_index] = 1
        elif playType == "test"	:
            Q_values = self.QNet.forward(torch.tensor([self.currentState], dtype=torch.float32)).detach().numpy()[0]
            action_index = np.argmax(Q_values)
            action[action_index] = 1

        return action
            
    def setInitState(self, observation):
        self.currentState = np.stack([observation for _ in range(self.config.multPerdInpt)], axis=0)

    def epsilonBuild(self):
        betta = 0.8
        if self.config.maxEpisodesTrain != 0:
            epsilon_red = (self.config.epsilonBeg - self.config.epsilonEnd) / (self.config.maxEpisodesTrain * betta)
        else:
            epsilon_red = 0
        return epsilon_red

    def epsilonReduce(self):
        if self.epsilon >self.config.epsilonEnd:
            self.epsilon -= self.epsilonRed
    
    def copyTargetQNetwork(self):
        self.target_QNet.load_state_dict(self.QNet.state_dict())

