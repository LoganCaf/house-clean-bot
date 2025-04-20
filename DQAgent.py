## DQAgent.py
# This file contains the DQAgent class, which implements a Deep Q-Learning agent using TensorFlow and Keras.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
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
        self.sampleSize = 256
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
        self.targetModel.save_weights(f"models/Model-{datetime.now().strftime("%d-%m-%Y-%H-%M")}.weights.h5")
        self.actionModel.save_weights(f"models/Model-latest.weights.h5")

    def reset(self):
        pass

    def buildModel(self):
        model = Sequential([
        Conv2D(64, (3,3), activation='relu', padding='same', input_shape=self.inputShape),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(self.outputShape, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learningRate), loss='mse')
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
        states = np.array([b[0] for b in batch]).reshape((self.sampleSize, *self.inputShape))
        currTargets = self.actionModel.predict(states, verbose=0, batch_size=self.sampleSize)
        nextTargets = self.targetModel.predict(np.array([b[3] for b in batch]).reshape((self.sampleSize, *self.inputShape)), verbose=0, batch_size=self.sampleSize)

        for i, (state, action, reward, nextState, done) in enumerate(batch):
            if done:
                currTargets[i][action] = reward
            else:
                currTargets[i][action] = reward + self.gamma * np.max(nextTargets[i])
        self.actionModel.fit(states, currTargets, epochs=1, verbose=0, batch_size=self.sampleSize)
        self.updateTargetModel()

