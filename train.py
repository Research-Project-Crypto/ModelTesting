import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from jupyterplot import ProgressPlot
import torch.nn as nn
import copy
import time
from itertools import count
import math
import gc

gc.enable()

# from pympler.tracker import SummaryTracker
# tracker = SummaryTracker()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, n_actions, feature_size, hidden_size=128):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.lstm_1 = nn.LSTM(input_size = feature_size, hidden_size = hidden_size, num_layers = 1, dropout = 0.1)
        self.output_layer = nn.Linear(hidden_size, n_actions)
    
    def forward(self, observation, action, hidden = None):
        lstm_input = observation
        if hidden is not None:
            # print('hidden not None')
            # print(observation.shape)
            # print(action.shape)
            lstm_out, hidden_out = self.lstm_1(lstm_input, hidden)
        else:
            # print('hidden None')
            # print(observation.shape)
            # print(action.shape)
            lstm_out, hidden_out = self.lstm_1(lstm_input)
        q_values = self.output_layer(lstm_out)
        return q_values, hidden_out
    
    def predict(self, observation, last_action, epsilon, hidden = None):
        q_values, hidden_out = self.forward(observation, last_action, hidden)
        if np.random.uniform() > epsilon:
            action = torch.argmax(q_values[0][0][-1]).item()
        else:
            action = np.random.randint(self.n_actions)
        return action, hidden_out

class ExpBuffer():
    def __init__(self, max_storage, sample_length):
        self.max_storage = max_storage
        self.sample_length = sample_length
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]

    def write_tuple(self, aoarod):
        if self.counter < self.max_storage-1:
            self.counter +=1
        if self.filled < self.max_storage:
            self.filled += 1
        else:
            self.counter = 0
        self.storage[self.counter] = aoarod
    
    def sample(self, batch_size):
        #Returns sizes of (batch_size, seq_len, *) depending on action/observation/return/done
        seq_len = self.sample_length
        last_actions = []
        last_observations = []
        actions = []
        rewards = []
        observations = []
        dones = []

        for i in range(batch_size):
            if self.filled - seq_len < 0 :
                raise Exception("Reduce seq_len or increase exploration at start.")
            start_idx = np.random.randint(self.filled-seq_len)
            #print(self.filled)
            #print(start_idx)
            last_action, last_observation, action, reward, observation, done = zip(*self.storage[start_idx:start_idx+seq_len])
            
            # print(len(last_observation[0]))
            last_actions.append(list(last_action))
            last_observations.append(last_observation)
            actions.append(list(action))
            rewards.append(list(reward))
            observations.append(list(observation))
            dones.append(list(done))
           
        return torch.tensor(last_actions).to(device), torch.tensor(last_observations, dtype = torch.float32).to(device), torch.tensor(actions).to(device), torch.tensor(rewards).float().to(device) , torch.tensor(observations, dtype = torch.float32).to(device), torch.tensor(dones).to(device)

# actions [0,1,2]
# 0 = do nothing
# 1 = buy
# 2 = sell

class CryptoEnv:
    _actions = [0,1,2]
    _frame_pos = 0
    _pos = -1
    
    def __init__(self, filename, frame_size = 128):
        self._frame_size = frame_size
        
        self._filename = filename
        self.reset(filename)
        
    def __calc_reward(self, sell_pos):
        return (sell_pos - self._pos) / sell_pos * 100
    
    def __init_values(self, df : pd.DataFrame):
        mask = [ 'maxprofitclose', 'maxprofitlowhigh' ]
        
        targets = df[mask].values.tolist()
        
        mask.append('event_time')
        features = df.drop(columns = mask).values.tolist()
        
        return features, targets

    def step(self, action = None):
        if action not in self._actions:
            raise ValueError("Chosen action is not a valid one.")
        
        self._frame_pos += 1
        
        if self._data_length < self._frame_pos + self._frame_size:
            return [], 0, True
        
        new_state = self._features[self._frame_pos: self._frame_pos + self._frame_size]

        if self._pos != -1 and action == 2:
            reward = self.__calc_reward(self._targets[self._frame_pos + self._frame_size - 1][0])
            # print(f"sell on {self._frame_pos + self._frame_size}: {reward}")
            self._pos = -1
            
            return new_state, reward, False
        
        if self._pos == -1 and action == 1:
            self._pos = self._targets[self._frame_pos + self._frame_size - 1][0]      
            # print(f"buy on {self._frame_pos + self._frame_size}")
            return new_state, 0, False
        
        # print(f"hodl")
        return new_state, 0, False
            
        
    def reset(self, filename = None):
        print("reset")
        if filename == None:
            filename = self._filename
        df = pd.read_csv(filename)
        
        self._features, self._targets = self.__init_values(df)
        
        del(df)
        
        self._data_length = len(self._features)
        self._frame_pos = 0
        self._pos = -1
        
        return self._features[self._frame_pos: self._frame_pos + self._frame_size]

frame_size = 50

env = CryptoEnv('./data/AAVEUSDT.csv', frame_size)

feature_size = len(env._features[0])
n_actions = len(env._actions)

M_episodes = 5
replay_buffer_size = 100
sample_length = 20
replay_buffer = ExpBuffer(replay_buffer_size, sample_length)
batch_size = 64
eps_start = 0.9
eps = eps_start
eps_end = 0.05
eps_decay = 0.99999
gamma = 0.999
learning_rate = 0.01
blind_prob = 0
EXPLORE = 0

print("feature_size:", feature_size)
print("n_actions:", n_actions)

pp = ProgressPlot(plot_names = ['Return', 'Exploration'], line_names = ['Value'])
# dqn = DQN(n_actions, state_size, embedding_size).cuda()
# adrqn_target = DQN(n_actions, state_size, embedding_size).cuda()
dqn = DQN(n_actions, feature_size).to(device)
dqn_target = DQN(n_actions, feature_size).to(device)
dqn_target.load_state_dict(dqn.state_dict())

optimizer = torch.optim.Adam(dqn.parameters(), lr = learning_rate)

for i_episode in range(M_episodes):
    print(f"new episode {i_episode}")
    done = False
    hidden = None
    last_action = 0
    current_return = 0
    
    last_observation = env.reset()
    
    for t in count():
        if t % 10000 == 0:
            print(t)
            # tracker.print_diff()
            gc.collect(generation=2)
            print(gc.garbage)

        observation_tensors = torch.tensor(last_observation).float().view(1, env._frame_size, feature_size).to(device)
        last_action_tensors = F.one_hot(torch.tensor(last_action), n_actions).view(1,1,-1).float().to(device)

        action, hidden = dqn.predict(
            observation_tensors,
            last_action_tensors,
            hidden = hidden,
            epsilon = eps
        )

        observation, reward, done = env.step(action)
        # if np.random.rand() < blind_prob:
        #     #Induce partial observability
        #     observation = np.zeros_like(observation)

        reward = np.sign(reward)
        current_return += reward
        replay_buffer.write_tuple((last_action, last_observation[-1], action, reward, observation[-1], done))
        
        last_action = action
        last_observation = observation

        
    
        #Updating Networks
        if i_episode > EXPLORE:
                # eps = eps_end + (eps_start - eps_end) * math.exp((-1*(i_episode-EXPLORE))/eps_decay)
                eps = eps * eps_decay

                last_actions, last_observations, actions, rewards, observations, dones = replay_buffer.sample(batch_size)
                # print(last_observations.shape)
                q_values, _ = dqn.forward(last_observations, F.one_hot(last_actions, n_actions).float())
                q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
                predicted_q_values, _ = dqn_target.forward(observations, F.one_hot(actions, n_actions).float())
                target_values = rewards + (gamma * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])

                #Update network parameters
                optimizer.zero_grad()
                loss = torch.nn.MSELoss()(q_values , target_values.detach())
                loss.backward()
                optimizer.step()      
        if done:
            break

    pp.update([[current_return],[eps]])
    dqn_target.load_state_dict(dqn.state_dict())

env.close()

torch.save(dqn.state_dict(), "models/")






































