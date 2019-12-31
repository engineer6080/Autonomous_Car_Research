from time import time
import numpy as np
import copy
from collections import namedtuple, deque

from Actor import Actor
from Critic import Critic
#from Noise import OUNoise

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.utils import to_categorical
from keras import optimizers
import tensorflow as tf
import keras
import cv2
import random as rn

import numpy as np
import copy


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class DDPG():
	"""Reinforcement Learning agent that learns using DDPG."""
	def __init__(self, state_size, action_size, train=True):

		self.train = train
		self.action_size = action_size 
		self.state_size = state_size

		actor_lr =  0.001    #Learning rate for Actor 0.0001
		critic_lr = 0.01     #Lerning rate for Critic 0.001

		deep_lr = 1e-3
		
		# Noise process
		self.exploration_mu = 0 # Mean
		self.exploration_theta = 0.6 #.15 How fast variable reverts to mean
		self.exploration_sigma = 0.3 # .2 Degree of volatility
		self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)


		if(self.train):
			# Actor (Policy) Model
			self.actor_local = Actor(self.state_size, self.action_size, actor_lr)
			self.actor_target = Actor(self.state_size,  self.action_size, actor_lr)

			# Critic (Value) Model
			self.critic_local = Critic(self.state_size, self.action_size, critic_lr)
			self.critic_target = Critic(self.state_size, self.action_size, critic_lr)

			# Initialize target model parameters with local model parameters
			self.critic_target.model.set_weights(self.critic_local.model.get_weights())
			self.actor_target.model.set_weights(self.actor_local.model.get_weights())

			# Replay memory
			self.buffer_size = 300 #1024
			self.batch_size = 32 #32

			# internal memory (deque)
			self.memory = deque(maxlen=self.buffer_size)
			#self.memory = []
			self.experience = namedtuple("Data", field_names=["state", "action", "reward", "next_state", "done"])

			# Algorithm parameters
			self.gamma = 0.99  # discount factor
			self.tau = 0.01   # for soft update of target parameters 0.001

			self.guide = False

			print("DDPG init", "Actor: ", actor_lr, "Critic: ", critic_lr)
			#print("Tau: ", self.tau, "Sigma: ", self.exploration_sigma)
			print(self.actor_local.model.summary())
			print(self.critic_local.model.summary())

			self.batch_id = 0
			self.critic_loss = 0
			self.actor_loss = 0

			self.C_loss= []
			self.A_loss = []
			
	def save_model(self, num):
		# Save the weights weights-improvement--0.03.hdf5
		load_str = "weights-improvement--0.{}.hdf5".format(num)
		self.deep_NN.model.load_weights(load_str)
		self.deep_NN.model.save("./model/model.h5")
		print("Saved model with best weights to disk")

	def load_model(self, name):
		# Save the weights
		self.deep_NN.model.load_weights(name)

	def summarize_prediction(self,Y_true, Y_pred):
		mse = mean_squared_error(Y_true, Y_pred)
		r_squared = r2_score(Y_true, Y_pred)
		print("mse       = {0:.2f}".format(mse))
		print("r_squared = {0:.2f}%".format(r_squared))

	def predict_and_summarize(self,X, Y):
		model = load_model("./model/model.h5")
		Y_pred = model.predict(X).astype('int')
		self.summarize_prediction(Y, Y_pred)
		return Y_pred

	def predict(self, state):
		"""Returns actions for given state(s) as per current policy."""
		#state = np.reshape(state, [-1, self.state_size])
		#action = self.trained.model.predict(state)[0]
		noise = self.noise.sample()
		action = self.actor_target.model.predict(state)
		return action, noise

	def get_sample(self, b_size=None):
		if(b_size is None):
			b_size = self.batch_size
		return rn.sample(self.memory, k=b_size)

	def conv_to_tensor(self, img):
		# Black and White Image ex: 1, 244, 244, 1
		if(len(img.shape) == 2):
			img = np.expand_dims(img, axis=3)
			img = np.expand_dims(img, axis=0)
		# RGB Image or stacked image: 1, 244, 244, 3
		elif(len(img.shape) == 3):
			img = np.expand_dims(img, axis=0)
		return img

	def reset(self):
		self.critic_loss = 0
		self.memory.clear()
		
	def step(self, state, action, reward, next_state, done):
			d = self.experience(state, action, reward, next_state, done)
			if(len(self.memory) == self.buffer_size):
				self.memory.popleft()
			self.memory.append(d)

	def learn(self, verbose=False): #experiences 
		"""Update policy and value parameters using given batch of experience tuples."""
		if(len(self.memory) < self.batch_size):
			return 
		
		experiences = self.get_sample()
		
		if(verbose):
			print("Buffer Size: ", len(self.memory))
			print("Sample Size: ", len(experiences))

		# Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
		states = np.vstack([self.conv_to_tensor(e.state) for e in experiences if e is not None])
		actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
		rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
		dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
		next_states = np.vstack([self.conv_to_tensor(e.next_state) for e in experiences if e is not None])

		if(0):
			print("States", states.shape)
			print("Actions", actions.shape)
			print("Rewards", rewards.shape)
			print("Next States", next_states.shape)
			print("Dones", dones.shape)

		# keep training actor local and critic local
		# use values from target model to update and train local
		# don't train target models, we soft update target

		actions_next = self.actor_target.model.predict_on_batch(next_states)

		#print("Actions next", actions_next.shape)

		Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

		Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

		self.critic_loss = self.critic_local.model.train_on_batch(x=[states,actions], y=Q_targets)

		# Train actor model (local)
		action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
		self.actor_loss = self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

		self.A_loss.append(self.actor_loss)
		self.C_loss.append(self.critic_loss)
		#self.batch_id += 1

		# Soft-update target models
		self.soft_update(self.critic_local.model, self.critic_target.model)
		self.soft_update(self.actor_local.model, self.actor_target.model)   

	def soft_update(self, local_model, target_model):
		"""Soft update model parameters."""
		local_weights = np.array(local_model.get_weights())
		target_weights = np.array(target_model.get_weights())

		assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

		new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
		target_model.set_weights(new_weights)