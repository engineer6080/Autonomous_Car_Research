from keras import layers, models, optimizers, regularizers
from keras import backend as K
from keras.layers import Flatten, Concatenate, LeakyReLU
from keras.utils.generic_utils import get_custom_objects
from keras.initializers import RandomUniform, Zeros
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import numpy as np

class Critic:
	"""Critic (Value) Model."""

	def __init__(self, state_size, action_size, lr):
		"""Initialize parameters and build model.

		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
		"""
		self.state_size = state_size
		self.action_size = action_size
		self.lr = lr
		self.build_model()

	def build_model(self):
		"""Build a critic (value) network that maps (state, action) pairs -> Q-values."""

		# Define input layers
		actions = layers.Input(shape=(self.action_size,), name='actions')        
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
		statesactions = Concatenate()([l3, actions]) 

		l4 = layers.Dense(50, activation='elu')(statesactions)
		l5 = layers.Dense(10, activation='elu')(l4)

		# Combine state and action pathways
		# Add final output layer to produce action values (Q values)
		Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=RandomUniform())(l5)

		# Create Keras model
		self.model = models.Model(inputs=[states, actions], outputs=Q_values)

		# Define optimizer and compile model for training with built-in loss function
		optimizer = optimizers.Adam(self.lr)
		self.model.compile(optimizer=optimizer, loss='mse')

		# Compute action gradients (derivative of Q values w.r.t. to actions)
		action_gradients = K.gradients(Q_values, actions)

		# Define an additional function to fetch action gradients (to be used by actor model)
		self.get_action_gradients = K.function(
			inputs=[*self.model.input, K.learning_phase()],
			outputs=action_gradients)
		
	# Numerical state input
	def build_model_old(self):
		
		# Define input layers
		actions = layers.Input(shape=(self.action_size,), name='actions')        
		states = layers.Input(shape=(self.state_size), name='input')
		    
		w1 = layers.Dense(64, activation='elu')(states)
		a1 = layers.Dense(64, activation='elu')(actions)
		#h1 = layers.Dense(30, activation='linear')(w1)
		h2 = Concatenate()([w1,a1]) 
		h3 = layers.Dense(64, activation='elu')(h2)
		
		# Combine state and action pathways
		# Add final output layer to produce action values (Q values)
		Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=RandomUniform())(h3)

		# Create Keras model
		self.model = models.Model(inputs=[states, actions], outputs=Q_values)

		# Define optimizer and compile model for training with built-in loss function
		optimizer = optimizers.Adam(self.lr)
		self.model.compile(optimizer=optimizer, loss='mse')

		# Compute action gradients (derivative of Q values w.r.t. to actions)
		action_gradients = K.gradients(Q_values, actions)

		# Define an additional function to fetch action gradients (to be used by actor model)
		self.get_action_gradients = K.function(
			inputs=[*self.model.input, K.learning_phase()],
			outputs=action_gradients)