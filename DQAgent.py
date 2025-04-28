## DQAgent.py
# This file contains the DQAgent class, which implements a Deep Q-Learning agent using TensorFlow and Keras.

import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Lambda, Add, Reshape, Concatenate, Multiply, Softmax, Layer, Activation, Subtract
from tensorflow.keras.optimizers import Adam
import random
from datetime import datetime
from PER import PrioritizedReplayBuffer


class DQAgent:
    def __init__(self,inputShape,outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.epsilon = 1
        self.epsilonDecay = 0.99995
        self.learningRate = 0.0001
        self.epsilonMin = 0.01
        self.gamma = 0.99
        self.memory = PrioritizedReplayBuffer(capacity=1_000_000, alpha=0.6)
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
    
    def saveModel(self,addon=""):
        self.actionModel.save_weights(f"models/Model-{datetime.now().strftime('%d-%m-%Y-%H-%M')}-{addon}.weights.h5")
    
    def updateTargetModel(self):
        self.updateTime -=1
        if self.updateTime > 0:
            return
        self.targetModel.set_weights(self.actionModel.get_weights())
        self.updateTime = 1000
        if self.epsilon < .9:
            self.targetModel.save_weights(f"models/Model-{datetime.now().strftime('%d-%m-%Y-%H-%M')}.weights.h5")
            self.actionModel.save_weights(f"models/Model-latest.weights.h5")

    def reset(self):
        pass

    def buildModel(self):
        inputs = Input(shape=self.inputShape)

        # shared convolution trunk
        x = Conv2D(64, 3, activation='relu', padding="same")(inputs)
        x = Conv2D(64, 3, activation='relu', padding="same")(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)         # (batch, 128)

        # value stream
        v = Dense(128, activation='relu')(x)
        v = Dense(1)(v)             # V(s) in fp32

        # advantage stream
        a = Dense(128, activation='relu')(x)
        a = Dense(self.outputShape, activation='linear')(a)  # A(s,a)

        mean_a = Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
        q_values = tf.keras.layers.add([v, Lambda(lambda t: t[0] - t[1])([a, mean_a])])

        # compile
        model = Model(inputs, q_values, name="Dueling_DQN")
        model.compile(
            optimizer=Adam(self.learningRate),
            loss=tf.keras.losses.Huber(delta=1.0)           # robust for RL targets
        )
        return model

    # call to take an action in the environment
    def act(self,state):
        self.lastState = state.reshape((1, *self.inputShape)).copy()
        if np.random.rand() <= self.epsilon:
            self.lastAction = random.randrange(self.outputShape)
        else:
            act_values = self.actionModel.predict(self.lastState, verbose=0)
            self.lastAction = np.argmax(act_values[0])
        return self.lastAction
    
    # call to remember the last action taken and the reward received
    # this is called after the action has been taken and a new state has been received
    def remember(self,state,reward,done):
        self.memory.add((self.lastState, self.lastAction, reward,state.reshape((1, *self.inputShape)).copy(), done))
        self.train()
    
    def train(self):
        if len(self.memory) < self.minMemorySize:
            return
        self.epsilon *= self.epsilonDecay
        if self.epsilon < self.epsilonMin:
            self.epsilon = self.epsilonMin
        # if self.trainTime > 0:
        #     self.trainTime -= 1
        #     return
        # self.trainTime = 10
        idxs, batch, is_w = self.memory.sample(self.sampleSize)

        states      = np.vstack([exp[0] for exp in batch])           # shape: (batch, H, W, C)
        actions     = np.array([exp[1] for exp in batch])            # shape: (batch,)
        rewards     = np.array([exp[2] for exp in batch])            # shape: (batch,)
        next_states = np.vstack([exp[3] for exp in batch])           # shape: (batch, H, W, C)
        dones       = np.array([exp[4] for exp in batch])            # shape: (batch,)

        q_next_online = self.actionModel.predict(next_states, verbose=0)  # shape: (batch, n_actions)
        q_next_target = self.targetModel.predict(next_states, verbose=0)  # shape: (batch, n_actions)

        targets = self.actionModel.predict(states, verbose=0)
        q_current = targets.copy()

        targets[ np.arange(self.sampleSize), actions ] = (
            rewards + (1 - dones) * self.gamma *
            q_next_target[np.arange(self.sampleSize), q_next_online.argmax(1)]  # select via online net and evaluate via target net
        )

        td_errors = targets[np.arange(self.sampleSize), actions] - \
                    q_current[np.arange(self.sampleSize), actions]

        self.actionModel.fit(states, targets,sample_weight=is_w,batch_size=self.sampleSize,verbose=0)

        self.memory.update_priorities(idxs, td_errors)

        self.updateTargetModel()