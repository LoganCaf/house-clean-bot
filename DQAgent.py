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
        self.epsilonDecay = .99995
        self.learningRate = 0.025
        self.epsilonMin = 0.1
        self.gamma = 0.95
        self.memory = deque(maxlen=20480)
        self.minMemorySize = 1280
        self.sampleSize = 128
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
        self.targetModel.save_weights(f"Model-{datetime.now()}.weights.h5")

    def reset(self):
        pass

    def buildModel(self):
        model = Sequential()
        model.add(Conv2D(128, (5, 5), activation='relu', input_shape=self.inputShape, padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.outputShape, activation='linear'))
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
        batch = random.sample(self.memory, self.sampleSize*self.trainTime)
        states = np.array([b[0] for b in batch]).reshape((self.sampleSize*self.trainTime, *self.inputShape))
        currTargets = self.actionModel.predict(states, verbose=0, batch_size=self.sampleSize)
        nextTargets = self.targetModel.predict(np.array([b[3] for b in batch]).reshape((self.sampleSize*self.trainTime, *self.inputShape)), verbose=0, batch_size=self.sampleSize)

        for i, (state, action, reward, nextState, done) in enumerate(batch):
            if done:
                currTargets[i][action] = reward
            else:
                currTargets[i][action] = reward + self.gamma * np.argmax(nextTargets[i])
        self.actionModel.fit(states, currTargets, epochs=1, verbose=0, batch_size=self.sampleSize)
        self.updateTargetModel()

