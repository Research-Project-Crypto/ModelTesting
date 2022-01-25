!pip install jupyterplot
!pip install matplotlib
!pip install numpy
!pip install torch
!pip install pandas
!pip install tensorflow
!pip install scikit-learn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import torch.nn.functional as F

import keras

from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
# from sklearn.utils import shuffle
# from sklearn.utils import class_weight
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization,SpatialDropout1D,Bidirectional, Embedding, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

from jupyterplot import ProgressPlot
import copy
import time
from itertools import count
import math

import torch

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = 'false'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

tensorboard = TensorBoard(log_dir="./logs")

# class DQN(nn.Module):
#     def __init__(self, n_actions, feature_size, hidden_size=128):
#         super(DQN, self).__init__()
#         self.n_actions = n_actions
#         self.lstm_1 = nn.LSTM(input_size = feature_size, hidden_size = hidden_size, num_layers = 1, dropout = 0.1)
#         self.output_layer = nn.Linear(hidden_size, n_actions)
    
#     def forward(self, observation, action, hidden = None):
#         lstm_input = observation
#         if hidden is not None:
#             # print('hidden not None')
#             # print(observation.shape)
#             # print(action.shape)
#             lstm_out, hidden_out = self.lstm_1(lstm_input, hidden)
#         else:
#             # print('hidden None')
#             # print(observation.shape)
#             # print(action.shape)
#             lstm_out, hidden_out = self.lstm_1(lstm_input)
#         q_values = self.output_layer(lstm_out)
#         return q_values, hidden_out
    
#     def predict(self, observation, last_action, epsilon, hidden = None):
#         q_values, hidden_out = self.forward(observation, last_action, hidden)
#         if np.random.uniform() > epsilon:
#             action = torch.argmax(q_values[0][0][-1]).item()
#         else:
#             action = np.random.randint(self.n_actions)
#         return action, hidden_out

class DQN(tf.keras.Model):
    def __init__(self, n_actions, feature_size, frame_size, layers = 2, layer_sizes = [128, 128], dropouts = [0.1, 0], batchnormalizations = [0, 0], optimizer='adam'):
        super().__init__()
        self._n_actions = n_actions
        self._feature_size = feature_size
        self._frame_size = frame_size

        self._model = self.create_model(layers, layer_sizes, dropouts, batchnormalizations, optimizer)
    
    def create_model(self, layers, layer_sizes, dropouts, batchnormalizations, optimizer):
        model = Sequential()

        for i in range(layers):
            if i == 0:
                model.add(LSTM(units=layer_sizes[i], return_sequences = True, input_shape = (self._frame_size, self._feature_size)))
            elif i != layers:
                model.add(LSTM(units=layer_sizes[i], return_sequences = True))
            else:
                model.add(LSTM(units=layer_sizes[i]))

            if dropouts[i] > 0:
                model.add(Dropout(dropouts[i]))
            if batchnormalizations[i] == 1:
                model.add(BatchNormalization()) 
        
        model.add(Dense(units=self._n_actions))
        
        model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

        return model
    
    def forward(self, observation):
        q_values = self._model.predict(observation)
        # q_values = self._model.make_predict_function(observation)
        # print(q_values.all())
        return q_values
    
    def predict(self, observation, epsilon):
        q_values = self.forward(observation)
        if np.random.uniform() > epsilon:
            action = np.argmax(q_values[0][-1], axis=-1)
        else:
            action = np.random.randint(self._n_actions)
        return action
    
    def fit(self, observations, targets, batch_size):
        self._model.fit(observations, targets, batch_size=batch_size)

########################## perhapst try this to find a decent optimizer and weight initializer
# def create_model(optimizer='adam', dropout_rate = 0.0, kernel_initializer='uniform', neurons = 128, batch_size = 16):
#     model = Sequential()
#     model.add(Dense(neurons, input_dim=X_train.shape[1], kernel_initializer=kernel_initializer,activation=activation))
#     model.add(Dropout(dropout_rate))
#     model.add(BatchNormalization())
#     model.add(Dense(neurons, kernel_initializer=kernel_initializer,activation=activation))
#     model.add(Dropout(dropout_rate))
#     model.add(BatchNormalization())
#     model.add(Dense(neurons, kernel_initializer=kernel_initializer,activation=activation))
#     model.add(Dropout(dropout_rate))
#     model.add(BatchNormalization())
#     model.add(Dense(y_train.shape[1], kernel_initializer=kernel_initializer, activation='softmax'))

#     model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizer,metrics=['accuracy']) 
#     return model

# model = KerasClassifier(build_fn=create_model, batch_size=64, epochs =10)

# dropout_rate = [0.0, 0.2, 0.4]
# neurons = [128]
# init = ['uniform', 'lecun_uniform', 'normal']
# optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# epochs = [50] 
# batch_size = [64] 

# param_grid = dict(epochs=epochs, batch_size=batch_size,optimizer = optimizer, dropout_rate = dropout_rate,activation = activation, kernel_initializer=init, neurons=neurons)
# grid = GridSearchCV(estimator=model, param_grid = param_grid,verbose=3)
# grid_result = grid.fit(X_train, y_train) 
###############################

class ExpBuffer():
    def __init__(self, max_storage, sample_length):
        self.max_storage = max_storage
        self.sample_length = sample_length
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]

    def write_tuple(self, oarod):
        if self.counter < self.max_storage-1:
            self.counter +=1
        if self.filled < self.max_storage:
            self.filled += 1
        else:
            self.counter = 0
        self.storage[self.counter] = oarod
    
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
            last_observation, action, reward, observation, done = zip(*self.storage[start_idx:start_idx+seq_len])
            
            # print(len(last_observation[0]))
            last_observations.append(last_observation)
            actions.append(list(action))
            rewards.append(list(reward))
            observations.append(list(observation))
            dones.append(list(done))
           
        return last_observations, actions, rewards, observations, dones

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
        # print(action)
        if action not in self._actions:
            raise ValueError("Chosen action is not a valid one.")
        
        self._frame_pos += 1
        
        if self._data_length < self._frame_pos + self._frame_size:
            return [], 0, True
        
        # new_state = self._features[self._frame_pos: self._frame_pos + self._frame_size]
        new_state = np.reshape(self._features[self._frame_pos: self._frame_pos + self._frame_size], (-1, self._frame_size, self._featurelength))

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
        self._featurelength = len(self._features[0])
        self._frame_pos = 0
        self._pos = -1
        
        return np.reshape(self._features[self._frame_pos: self._frame_pos + self._frame_size], (-1, self._frame_size, self._featurelength))

frame_size = 50

env = CryptoEnv('./data/AAVEUSDT.csv', frame_size)

feature_size = len(env._features[0])
n_actions = len(env._actions)

M_episodes = 25
replay_buffer_size = 100
# sample_length = 20
replay_buffer = ExpBuffer(replay_buffer_size, env._frame_size)
batch_size = 500
eps_start = 0.9
eps = eps_start
eps_end = 0.05
eps_decay = 0.99999
gamma = 0.999
learning_rate = 0.01
blind_prob = 0
EXPLORE = 5

print("feature_size:", feature_size)
print("n_actions:", n_actions)

pp = ProgressPlot(plot_names = ['Return', 'Exploration'], line_names = ['Value'])
dqn = DQN(n_actions, feature_size, env._frame_size, layers = 1, layer_sizes = [14, 128], dropouts = [0.1, 0], batchnormalizations = [0, 0], optimizer='adam')
dqn_target = DQN(n_actions, feature_size, env._frame_size, layers = 1, layer_sizes = [14, 128], dropouts = [0.1, 0], batchnormalizations = [0, 0], optimizer='adam')

dqn_target.set_weights(dqn.get_weights()) 

encoder = OneHotEncoder()
encoder.fit([[0,0],[1,1],[2,2]])

for i_episode in range(M_episodes):
    print(f"new episode {i_episode}")
    done = False
    hidden = None
    last_action = 0
    current_return = 0
    
    last_observation = env.reset()
    
    #for t in count():
    for t in range(10000):
        if t % 1000 == 0:
            print(t)

        action = dqn.predict(
            last_observation,
            epsilon = eps
        )

        observation, reward, done = env.step(action)
        # if np.random.rand() < blind_prob:
        #     #Induce partial observability
        #     observation = np.zeros_like(observation)

        reward = np.sign(reward)
        current_return += reward
        replay_buffer.write_tuple((last_observation[-1], action, reward, observation[-1], done))
        
        last_observation = observation

        # NOT YET CHANGED FOR TENSORFLOW
        if i_episode > EXPLORE:
            # eps = eps_end + (eps_start - eps_end) * math.exp((-1*(i_episode-EXPLORE))/eps_decay)

            last_observations, actions, rewards, observations, dones = replay_buffer.sample(batch_size)
            q_values = dqn.forward(np.reshape(last_observations[0][-1], (-1, env._frame_size, env._featurelength)))
            # print(q_values)
            # np.argmax(actions, axis=1)
            predicted_q_values = dqn_target.forward(np.reshape(observations[0][-1], (-1, env._frame_size, env._featurelength)))
            target_values = np.argmax(predicted_q_values, axis = -1)[0]

            targets = np.zeros((target_values.size, n_actions))
            targets[np.arange(target_values.size),target_values] = 1
            dqn.fit(np.reshape(observations[0][-1], (-1, env._frame_size, env._featurelength)), np.reshape(targets, (-1, len(targets), n_actions)), batch_size)
        
        if done:
            break

    pp.update([[current_return],[eps]])
    dqn_target.set_weights(dqn.get_weights())
    
    eps = eps * eps_decay

env.close()