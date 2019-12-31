from keras import layers, models, optimizers, regularizers
from keras import backend as K
from keras.initializers import RandomUniform
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

import numpy as np

class Actor:
	"""Actor (Policy) Model."""

	def __init__(self, state_size, action_size, lr):

		self.state_size = state_size
		self.action_size = action_size

		# Initialize any other variables here
		self.lr = lr

		self.build_model()

	def build_model(self):
		"""Build an actor (policy) network that maps states -> actions."""
		# Define input layer (states)
		states = layers.Input(shape=(self.state_size), name='input')

		c1 = layers.Convolution2D(filters=24, kernel_size=5, strides=2, activation='elu')(states)
		c2 = layers.Convolution2D(filters=36, kernel_size=5, strides=2, activation='elu')(c1)
		c3 = layers.Convolution2D(filters=48, kernel_size=5, strides=2, activation='elu')(c2)
		c4 = layers.Convolution2D(filters=64, kernel_size=3, activation='elu')(c3)
		d1 = layers.Dropout(0.2)(c4)
		c5 = layers.Convolution2D(filters=64, kernel_size=3, activation='elu')(d1)

		l1 = layers.Flatten()(c5)
		l2 = layers.Dropout(0.2)(l1)

		l3 = layers.Dense(100, activation='elu')(l2)

		l4 = layers.Dense(50, activation='elu')(l3)
		l5 = layers.Dense(10, activation='elu')(l4)

		# Steering and Throttle
		# Add final output layer with sigmoid activation
		raw_actions = layers.Dense(units=self.action_size, name='raw_actions', activation='sigmoid')(l5)
		
		# Scale [0, 1] output for each action dimension to proper range
		actions = layers.Lambda(lambda x: (x * 180) + 0,name='actions')(raw_actions)

		# Create Keras model
		self.model = models.Model(inputs=[states], outputs=actions)  #actions

		# Define loss function using action value (Q value) gradients
		action_gradients = layers.Input(shape=(self.action_size,))
		loss = K.mean(-action_gradients * actions)

		# Define optimizer and training function
		optimizer = optimizers.Adam(self.lr)
		updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

		self.train_fn = K.function(
		inputs=[self.model.input, action_gradients, K.learning_phase()],
		outputs=loss,
		updates=updates_op)

		
	# Numerical state input
	def build_model_old(self):
		
		# Define input layers
		states = layers.Input(shape=(self.state_size), name='input')
		
		w1 = layers.Dense(64, activation='elu')(states)
		h1 = layers.Dense(64, activation='elu')(w1)
		h2 = layers.Dense(64, activation='elu')(h1)
		
		# Steering and Throttle
		# Add final output layer with sigmoid activation
		raw_actions = layers.Dense(units=self.action_size, name='raw_actions', activation='sigmoid')(h2)
		#actions = layers.Dense(units=self.action_size, name='raw_actions')(h2)
		
		# Scale [0, 1] output for each action dimension to proper range
		actions = layers.Lambda(lambda x: (x * 180) + 0,name='actions')(raw_actions)

		# Create Keras model
		self.model = models.Model(inputs=[states], outputs=actions)  #actions

		# Define loss function using action value (Q value) gradients
		action_gradients = layers.Input(shape=(self.action_size,))
		loss = K.mean(-action_gradients * actions)

		# Define optimizer and training function
		optimizer = optimizers.Adam(self.lr)
		updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

		self.train_fn = K.function(
		inputs=[self.model.input, action_gradients, K.learning_phase()],
		outputs=loss,
		updates=updates_op)