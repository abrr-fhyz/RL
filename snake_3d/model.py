import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feed-forward neural network for Q-value approximation.
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # For Q–values, there is no activation in the final layer.
        return self.fc3(x)

class DQNAgent:
    def __init__(self, env,
                 state_size=None,
                 action_size=None,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_decay=0.99999,
                 min_epsilon=0.01,
                 batch_size=32,
                 memory_size=2000,
                 model_filename="dqn_model.pth"):
        # Use environment’s spaces or optionally pass state_size and action_size.
        self.env = env
        self.state_size = env.observation_space.shape[0] if state_size is None else state_size
        self.action_size = env.action_space.n if action_size is None else action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model_filename = model_filename

        # Create an instance of our network and set up the optimizer and loss function.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # Store experience for replay training.
        self.memory.append((state, action, reward, next_state, done))
        return self.epsilon

    def choose_action(self, state, e_g_enabled=True):
        # Epsilon-greedy action selection.
        if e_g_enabled and (np.random.rand() < self.epsilon):
            return random.randrange(self.action_size)
        # Convert state to torch tensor.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()  # Set network to evaluation mode.
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()  # Return to training mode.
        return int(torch.argmax(q_values, dim=1).item())

    def replay(self):
        # Train using a batch sampled from replay memory (if enough samples stored).
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        # Prepare batches.
        states = torch.FloatTensor(np.array([entry[0] for entry in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([entry[1] for entry in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([entry[2] for entry in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([entry[3] for entry in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([entry[4] for entry in minibatch]).astype(np.float32)).unsqueeze(1).to(self.device)

        # Compute Q-values for current states.
        current_q = self.model(states).gather(1, actions)
        
        # Compute maximum Q-values for next states (detach so gradients are not backpropagated through next state predictions).
        with torch.no_grad():
            max_next_q = self.model(next_states).max(dim=1, keepdim=True)[0]
        
        # Compute target Q-values using the Bellman equation.
        target_q = rewards + (self.gamma * max_next_q * (1 - dones))
        
        # Compute loss.
        loss = self.loss_fn(current_q, target_q)
        
        # Backpropagation.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease epsilon (exploration rate) after each training batch.
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_filename, map_location=self.device))
        self.model.to(self.device)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_filename)
