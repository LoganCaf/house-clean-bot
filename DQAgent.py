## DQAgent.py
# This file contains the DQAgent class, which implements a Deep Q-Learning agent using TensorFlow and Keras.

import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Lambda, add
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from datetime import datetime






class DQAgent:
    def __init__(self,inputShape,outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.epsilon = 1
        self.epsilonDecay = 0.99995
        self.learningRate = 0.001
        self.epsilonMin = 0.01
        self.gamma = 0.95
        self.memory = deque(maxlen=51200)
        self.minMemorySize = 5120
        self.sampleSize = 64
        self.actionModel = self.buildModel()
        self.targetModel = self.buildModel()
        self.updateTime = 0
        self.trainTime = 0
        self.updateTargetModel()
    
    def loadModel(self,modelPath):
        self.actionModel.load_weights(modelPath)
        self.targetModel.load_weights(modelPath)
        self.updateTargetModel()
    
    def updateTargetModel(self):
        self.updateTime -=1
        if self.updateTime > 0:
            return
        self.targetModel.set_weights(self.actionModel.get_weights())
        self.updateTime = 1000
        if self.epsilon < .9:
            self.targetModel.save_weights(f"models/Model-{datetime.now().strftime("%d-%m-%Y-%H-%M")}.weights.h5")
            self.actionModel.save_weights(f"models/Model-latest.weights.h5")

    def reset(self):
        pass

    def buildModel(self):
        inputs = Input(shape=self.inputShape)
        x = Conv2D(128, (7,7), activation='relu')(inputs)
        x = Conv2D(64, (3,3), activation='relu')(x)
        flat = Flatten()(x)

        v = Dense(128, activation='relu')(flat)
        v = Dense(32, activation='relu')(v)
        v = Dense(1, activation='linear')(v)  # V(s)

        a = Dense(128, activation='relu')(flat)
        a = Dense(32, activation='relu')(flat)
        a = Dense(self.outputShape, activation='linear')(a)  # A(s,a)

        mean_a = Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
        q_values = add([v, Lambda(lambda t: t[0] - t[1])([a, mean_a])])

        # 5) Compile model
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learningRate),
                    loss='mse')
        return model

    # call to take an action in the environment
    def act(self,state):
        self.lastState = state.reshape((1, *self.inputShape))
        if np.random.rand() <= self.epsilon:
            self.lastAction = random.randrange(self.outputShape)
        else:
            act_values = self.actionModel.predict(self.lastState, verbose=0)
            self.lastAction = np.argmax(act_values[0])
        self.epsilon *= self.epsilonDecay
        if self.epsilon < self.epsilonMin:
            self.epsilon = self.epsilonMin
        return self.lastAction
    
    # call to remember the last action taken and the reward received
    # this is called after the action has been taken and a new state has been received
    def remember(self,state,reward,done):
        self.memory.append((self.lastState,self.lastAction,reward,state.reshape((1, *self.inputShape)),done))
        self.train()
    
    def train(self):
        if len(self.memory) < self.minMemorySize:
            return
        if self.trainTime > 0:
            self.trainTime -= 1
            return
        self.trainTime = 10
        batch = random.sample(self.memory, self.sampleSize)

        states      = np.vstack([exp[0] for exp in batch])           # shape: (batch, H, W, C)
        actions     = np.array([exp[1] for exp in batch])            # shape: (batch,)
        rewards     = np.array([exp[2] for exp in batch])            # shape: (batch,)
        next_states = np.vstack([exp[3] for exp in batch])           # shape: (batch, H, W, C)
        dones       = np.array([exp[4] for exp in batch])            # shape: (batch,)

        q_next_online = self.actionModel.predict(next_states,  verbose=0)  # shape: (batch, n_actions)
        q_next_target = self.targetModel.predict(next_states,  verbose=0)  # shape: (batch, n_actions)

        q_current = self.actionModel.predict(states, verbose=0)

        for i in range(self.sampleSize):
            if dones[i]:
                q_current[i, actions[i]] = rewards[i]
            else:
                best_next_action = np.argmax(q_next_online[i])                 # select via online net
                target_q_value   = q_next_target[i, best_next_action]          # evaluate via target net
                q_current[i, actions[i]] = rewards[i] + self.gamma * target_q_value

        self.actionModel.fit(states, q_current, epochs=1, batch_size=self.sampleSize, verbose=0)
        self.updateTargetModel()

