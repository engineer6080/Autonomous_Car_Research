from time import time
from collections import namedtuple, deque

from keras import layers, models, optimizers, regularizers
#from keras.coeDense, Dropout, Activation, Flatten
from keras import backend as K
from keras.initializers import RandomUniform
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.utils import to_categorical
from keras import optimizers
import tensorflow as tf
import keras
import cv2
import random as rn

import numpy as np
import copy

# Transform train_on_batch return value
# to dict expected by on_batch_end callback
def named_logs(model, logs):
	result = {}
	for l in zip(model.metrics_names, logs):
		result[l[0]] = l[1]
	return result


# Deep Q-learning Agent
class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = 0.99   
		self.epsilon = 0.2
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = .0001 #.0001
		self.tau = 0.1
		self.buffer_size = 2000
		self.model = self._build_model()
		self.target_model = self._build_model()
		print(self.model.summary())
		
		# internal memory (deque)
		self.memory = deque(maxlen=self.buffer_size)
		self.A_loss = []
		self.experience = namedtuple("Data", field_names=["state", "action", "reward", "next_state", "done"])
		
		# Create the TensorBoard callback,
		# which we will drive manually
		self.tensorboard = TensorBoard(
		  log_dir='/tmp/my_tf_logs',
		  histogram_freq=0,
		  batch_size=32,
		  write_graph=True,
		  write_grads=True
		)
		self.tensorboard.set_model(self.target_model)
		self.batch_id = 0
		
	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		# Define input layer (states)
		states = layers.Input(shape=(self.state_size), name='input')
		c1 = layers.Convolution2D(filters=32, kernel_size=8, strides=4, activation='relu')(states) # edge detection
		c2 = layers.Convolution2D(filters=64, kernel_size=4, strides=2, activation='relu')(c1)
		c3 = layers.Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu')(c2)
		l1 = layers.Flatten()(c3)
		l2 = layers.Dense(256, activation='relu')(l1)
		Q_val = layers.Dense(units=self.action_size, name='Q_Values', activation='linear')(l2)
		# Create Keras model
		model = models.Model(inputs=[states], outputs=Q_val)  #actions
		model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
		self.get_conv = K.function(
			inputs=[model.input],
			outputs=model.layers[1].output)
		return model
	
	def _build_model_old(self):
		# NVIDIA
		frame = layers.Input(shape=(self.state_size), name='input')

		c1 = layers.Convolution2D(filters=24, kernel_size=5, strides=2, activation='elu')(frame)
		c2 = layers.Convolution2D(filters=36, kernel_size=5, strides=2, activation='elu')(c1)
		c3 = layers.Convolution2D(filters=48, kernel_size=5, strides=2, activation='elu')(c2)
		c4 = layers.Convolution2D(filters=64, kernel_size=3, activation='elu')(c3)
		c5 = layers.Convolution2D(filters=64, kernel_size=3, activation='elu')(c4)

		l1 = layers.Flatten()(c5)
		l2 = layers.Dense(100, activation='elu')(l1)
		l3 = layers.Dense(50, activation='elu')(l2)
		
		Q_val = layers.Dense(units=self.action_size, name='Q_Values', activation='linear')(l3)

		model = models.Model(inputs=[frame], outputs=Q_val) 

		# we use MSE (Mean Squared Error) as loss function
		model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
		self.get_conv = K.function(
			inputs=[model.input],
			outputs=model.layers[1].output)
		return model

	def step(self, state, action, reward, next_state, done):
		d = self.experience(state, action, reward, next_state, done)
		if(len(self.memory) == self.buffer_size):
			self.memory.popleft()
		self.memory.append(d)
		
	def conv_to_tensor(self, img): ###CHANGE
		if(len(img)<10): # why is this here??
			return img
		# Black and White Image ex: 1, 244, 244, 1
		if(len(img.shape) == 2):
			img = np.expand_dims(img, axis=3)
			img = np.expand_dims(img, axis=0)
		# RGB Image or stacked image: 1, 244, 244, 3
		elif(len(img.shape) == 3):
			img = np.expand_dims(img, axis=0)
		return img
		
	def predict(self, state):
		if np.random.rand() <= self.epsilon:
			return rn.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action
	
	def learn(self, batch_size=32, target_train=False):
		if(len(self.memory) < batch_size):
			return 
		minibatch = rn.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				 # Updating value of best action taken at the moment
				target = reward + self.gamma * \
					   np.amax(self.target_model.predict(self.conv_to_tensor(next_state))[0])
			target_f = self.target_model.predict(self.conv_to_tensor(state))
			target_f[0][action] = target
			qloss = self.model.fit(self.conv_to_tensor(state), target_f, epochs=1, verbose=0)
			logs = qloss.history['loss'][0]
			self.tensorboard.on_epoch_end(self.batch_id, named_logs(self.target_model, [logs]))
			self.A_loss.append(qloss)
		self.batch_id += 1
		if(target_train):
			print("TARGET TRAIN")
			self.target_train()
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
			
	def target_train(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		self.target_model.set_weights(target_weights)
