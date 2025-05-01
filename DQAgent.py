## DQAgent.py
## Author: Logan Caffey
# This file contains the DQAgent class, which implements a Deep Q-Learning agent using TensorFlow and Keras.

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Lambda, MaxPooling2D
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
        self.memory = PrioritizedReplayBuffer(capacity=20000, alpha=0.6)
        self.minMemorySize = 5120
        self.sampleSize = 64
        self.actionModel = self.buildModel()
        self.targetModel = self.buildModel()
        self.updateTime = 0
        self.trainTime = 0
        self.updateTargetModel()
    
    ## load the model from a file
    def loadModel(self,modelPath):
        self.actionModel.load_weights(modelPath)
        self.targetModel.load_weights(modelPath)
        self.updateTargetModel()
    
    ## save the model to a file
    def saveModel(self,addon=""):
        self.actionModel.save_weights(f"models/Model-{datetime.now().strftime('%d-%m-%Y-%H-%M')}-{addon}.weights.h5")
    
    ## update target model from action model
    def updateTargetModel(self):
        self.updateTime -=1
        if self.updateTime > 0: # only update after timer
            return
        self.targetModel.set_weights(self.actionModel.get_weights())
        self.updateTime = 1000
        if self.epsilon < .9: # only save trained for a bit
            self.targetModel.save_weights(f"models/Model-{datetime.now().strftime('%d-%m-%Y-%H-%M')}.weights.h5")
            self.actionModel.save_weights(f"models/Model-latest.weights.h5")

    ## build the model
    def buildModel(self):
        inputs = Input(shape=self.inputShape,dtype=tf.float32)

        # shared convolution trunk
        x = Conv2D(64, 3, activation='relu', padding="same")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, 3, activation='relu', padding="same")(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)

        # value stream
        v = Dense(128, activation='relu')(x)
        v = Dense(1)(v)

        # advantage stream
        a = Dense(128, activation='relu')(x)
        a = Dense(self.outputShape, activation='linear')(a)

        mean_a = Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
        q_values = tf.keras.layers.add([v, Lambda(lambda t: t[0] - t[1])([a, mean_a])])

        # compile
        model = Model(inputs, q_values, name="Dueling_DQN")
        model.compile(
            optimizer=Adam(self.learningRate),
            loss=tf.keras.losses.Huber(delta=1.0)
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
        if self.learningRate == 0:
            return
        self.memory.add((self.lastState, self.lastAction, reward,state.reshape((1, *self.inputShape)).copy(), done))
        self.train()
    
    # trains the action model using the experience replay buffer
    def train(self):
        if len(self.memory) < self.minMemorySize:
            return
        self.epsilon *= self.epsilonDecay
        if self.epsilon < self.epsilonMin:
            self.epsilon = self.epsilonMin
        idxs, batch, is_w = self.memory.sample(self.sampleSize)

        states      = np.vstack([exp[0] for exp in batch])
        actions     = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards     = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.vstack([exp[3] for exp in batch])
        dones       = np.array([exp[4] for exp in batch], dtype=np.float32)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        is_w = tf.convert_to_tensor(is_w, dtype=tf.float32)

        # Calculate Target Q-values
        # Double DQN: Use action_model to select best next action, target_model to evaluate it
        q_next_online_actions = tf.argmax(self.actionModel(next_states), axis=1, output_type=tf.int32) # Get best actions from online model
        q_next_target_values = self.targetModel(next_states) # Get all Q-values from target model

        # Select the Q-value from target network corresponding to the action selected by the online network
        action_indices = tf.stack([tf.range(self.sampleSize, dtype=tf.int32), q_next_online_actions], axis=1)
        q_next_selected = tf.gather_nd(q_next_target_values, action_indices)

        # Calculate target: R + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
        target_q_values = rewards + (1.0 - dones) * self.gamma * q_next_selected

        # Calculate Loss and Update Action Model
        with tf.GradientTape() as tape:
            # Get Q-values for the *actual actions taken* in the sampled states
            q_current_online_values = self.actionModel(states)
            action_indices_current = tf.stack([tf.range(self.sampleSize, dtype=tf.int32), actions], axis=1)
            q_current_selected = tf.gather_nd(q_current_online_values, action_indices_current)

            # Calculate TD Error for PER update
            td_errors = target_q_values - q_current_selected

            # Calculate Huber loss, weighted by importance sampling weights
            loss = self.actionModel.loss(target_q_values, q_current_selected)
            weighted_loss = tf.reduce_mean(loss * is_w)

        # Compute gradients and apply updates
        gradients = tape.gradient(weighted_loss, self.actionModel.trainable_variables)
        self.actionModel.optimizer.apply_gradients(zip(gradients, self.actionModel.trainable_variables))

        # Update PER priorities
        self.memory.update_priorities(idxs, tf.abs(td_errors).numpy())

        self.updateTargetModel()

        self.updateTargetModel()