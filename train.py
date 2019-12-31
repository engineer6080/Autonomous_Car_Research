import cv2
import time
from collections import namedtuple, deque

from approxeng.input.selectbinder import ControllerResource
import pickle
from camera import Camera
import numpy as np
import os

import serial
time.sleep(0.5)

# Storage for user training
# Need to store Pic, Steering, Throttle

class Train():
	def __init__(self, buffer_size=10000, width=244, height=244):
		self.memory = [] # list
		self.width = width
		self.height = height

		self.camera = Camera(width=width, height=height, capture_width=1280, capture_height=720, capture_fps=60, capture_flip=2)
		self.THROTTLE = 0
		self.STEERING = 0
		self.last_state = None

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)

	def getReward(self):
		return 1

	def save(self, name, vid=False):
		out_str = "./training_dat/{}.pkl".format(name)
		afile = open(out_str, 'wb')
		pickle.dump(self.memory, afile)
		afile.close()

	def controller(self):
		print("Running")
		START = 0
		IDLE = 1
		PAUSE = 2
		EXIT = 3
		STATE = PAUSE
		func_timer = 0

		with serial.Serial('/dev/ttyUSB0', 115200, timeout=10) as ser:
			with ControllerResource() as joystick:
				while joystick.connected:
					axis_list = [ 'ly', 'rx', 'cross','select', 'square'] # ACTUALLY LY, RX, SQUARE, l press, Left top button
					self.THROTTLE = int((joystick['ly']+1)/2*180)
					self.STEERING = int((joystick['rx']+1)/2*180)

					if(joystick['cross'] is not None and STATE != EXIT):  # SQUARE EXIT
						print("EXIT")
						ser.close()
						time_taken = time.time() - func_timer
						fps = len(training_module.memory)/time_taken

						print("Total time: ", time_taken)
						print("Frames: ", len(training_module.memory))
						print("FPS: ", fps)
						self.camera.unobserve_all()
						STATE = EXIT
						self.camera.stop()
						break
					if(joystick['square'] is not None and STATE != START): # L1 START
						print("START")
						self.camera.observe(update_image, names='value')
						func_timer = time.time()
						STATE = START

					output = "{:05d}-{:05d}\n".format(int(self.THROTTLE), int(self.STEERING))
					ser.write(bytes(output,'utf-8'))
					#time.sleep(0.016) # 16 ms = 60fps
					#time.sleep(0.032) # 30 fps

	def start(self):
		self.camera.unobserve_all()
		self.controller()

def update_image(change):
		image = change['new']
		next_state = image.copy()
		reward = training_module.getReward()
		action = [training_module.THROTTLE, training_module.STEERING]
		if(training_module.last_state is not None):
			d = [[training_module.last_state], action, reward, [next_state], 0]
			training_module.memory.append(d)
		training_module.last_state = next_state

training_module = Train()
print("L1 START")
print("SQUARE EXIT")

print("Enter filename or q to quit")
user = input()
if(user == 'q'):
	print("Goodbye")
	exit()
else:
	training_module.memory.clear()
	training_module.start()
	print("Saving..")
	training_module.save(str(user))
	print("Done!")
print("Goodbye")
