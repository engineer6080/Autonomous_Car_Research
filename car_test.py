from approxeng.input.selectbinder import ControllerResource
import serial

# PI Cam 160 FOV 60 FPS @ 1280x720 (GSTREAMER SAYS 120?)
with serial.Serial('/dev/ttyUSB0', 1000000, timeout=1) as ser:
	with ControllerResource() as joystick:

		# Initialize
		STEERING = 90
		THROTTLE = 90
		speed = 0
		while joystick.connected:
			if(joystick['cross'] is not None):  # SQUARE EXIT
				print("EXIT")
				output = "{:05d}-{:05d}\n".format(int(90), int(90))
				ser.write(bytes(output,'utf-8'))
				ser.close()
				break

			# Some manual 
			STEERING = int((joystick['rx']+1)/2*180)
			THROTTLE = int((joystick['ly']+1)/2*180)

			output = "{:05d}-{:05d}\n".format(int(THROTTLE), int(STEERING))
			ser.write(bytes(output,'utf-8'))
			#time.sleep(0.001)
			speed_in = (ser.readline())
			if speed_in:
				speed = speed_in

			print(speed)