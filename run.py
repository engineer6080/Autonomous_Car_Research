from jetcam.csi_camera import CSICamera
from tensorflow.keras.preprocessing import image
from approxeng.input.selectbinder import ControllerResource
from FindLane import findlane, img_preprocess
import numpy as np
import cv2
import time
import serial

WIDTH = 244
HEIGHT = 244
camera = CSICamera(width=WIDTH, height=HEIGHT, capture_width=1080, capture_height=720, capture_fps=120)
#camera = cv2.VideoCapture(0)
camera.running = True

#output_names = ['dense_11/Softmax']
output_names = ['dense_12/BiasAdd']
input_names = ['input']

import tensorflow as tf

def get_frozen_graph(graph_file):
	"""Read Frozen Graph file from disk."""
	with tf.gfile.FastGFile(graph_file, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	return graph_def

trt_graph = get_frozen_graph('./model/trt_graph.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

# Get graph input size
for node in trt_graph.node:
	if 'input' in node.name:
		size = node.attr['shape'].shape
		image_size = [size.dim[i].size for i in range(1, 4)]
		break
print("image_size: {}".format(image_size))

# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

# Optional image to test model prediction.
#img_path = 'ColorSample.jpg'

#img = image.load_img(img_path, target_size=image_size[:2])
#x = image.img_to_array(img)
#x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
#x = np.expand_dims(x, axis=0)
#x = np.expand_dims(x, axis=3)
#(x.shape)


with serial.Serial('/dev/ttyUSB0', 115200, timeout=10) as ser:
	with ControllerResource() as joystick:
		while joystick.connected:
			image = camera.value
			#image = cv2.resize(image,(244,244))
			x = img_preprocess(image)

			feed_dict = {
				input_tensor_name: x
			}

			STEERING = tf_sess.run(output_tensor, feed_dict)

			# decode the results into a list of tuples (class, description, probability)
			# (one such list for each sample in the batch)
			#out = np.argmax((preds).astype('int'))

			if(joystick['square'] is not None): # hold L1 START for human control
				STEERING = int((joystick['rx']+1)/2*180)

			THROTTLE = int((joystick['ly']+1)/2*180)
					
			output = "{:05d}-{:05d}\n".format(int(THROTTLE), int(STEERING))
			ser.write(bytes(output,'utf-8'))

			if(joystick['cross'] is not None):  # SQUARE EXIT
				ser.close()
				break

			#print('Predicted:', output)

exit()