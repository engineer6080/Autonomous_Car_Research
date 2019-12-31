import math
import cv2
import numpy as np

class CV_IMG_PROCESSOR():
	def __init__(self):
		self.preview_img = None
		self.raw_img = None
		self.line = None
		# Params
		self.canny_min = 160 #160
		self.area_min = 80 #80
		self.arc_length_min = 160 #140
		# PD stuff
		self.CONST_TIME = 0.017
        self.P = 4
        self.D = 0.03
        self.error_old = 0
        
	def getSteering(self, img, im_out=None):        
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(img_gray,self.canny_min,255)    
		contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #CHAIN_APPROX_SIMPLE, RETR_TREE, RETR_EXTERNAL

		leftBox = None
		rightBox = None
		slopeLeft = 0
		slopeRight = 0

		rows,cols = img.shape[:2]

		img2 = img.copy()
		# Debugging and visual
		if(im_out == "canny"):
			img2 = edges
		elif(im_out == "gray"):
			img2 = img_gray
		elif(im_out == "black"):
			img2 = np.zeros([height,width,3],dtype=np.uint8)
		elif(im_out == "contours"): # All contours
			cv2.drawContours(img2, contours, -1, (0,0,255), 2)		

		DRAW_BOXES = 0
		LEFT_LINE = False
		RIGHT_LINE = False

		for c in contours:
			rr = cv2.minAreaRect(c)
			wh = rr[1]
			area = max(wh)
			#area = cv2.contourArea(c)
			length = cv2.arcLength(c,True)

			if(length > self.arc_length_min and area > self.area_min):
				#cv2.drawContours(img2, [c], -1, (255,0,0), 2)
				epsilon = 0.008*cv2.arcLength(c,True)
				approx = cv2.approxPolyDP(c,epsilon,True)

				box = cv2.boxPoints(rr)
				box = np.int0(box)
				box_X = box[0][0]        
				box_Y = box[0][1]   

				# Bottom left corner
				bl = np.where(box[:,0] == min(box[:,0]))
				# Top right ocrner
				tr = np.where(box[:,0] == max(box[:,0]))
				# Most of the time ?
				bottomLeft =  box[bl][0]
				topRight = box[tr][0]

				try:
					slope = (topRight[1]-bottomLeft[1])/(topRight[0]-bottomLeft[0])*-1				
				except:
					slope = 1

				if(box_X < 100 and box_Y > 100):				
					# Left Box
					if(DRAW_BOXES):
						cv2.drawContours(img2, [box], -1, (0,0,255), 2)

					if(leftBox is not None):
						if(cv2.arcLength(approx,True)>cv2.arcLength(leftBox,True)):
							leftBox = approx
							slopeLeft = slope
					else:
						LEFT_LINE = True
						leftBox = approx
						slopeLeft = slope

				elif(box_X > 144 and box_Y > 100):
					# Right Box
					if(DRAW_BOXES):
						cv2.drawContours(img2, [box], -1, (255,0,0), 2)

					if(rightBox is not None):
						if(cv2.arcLength(approx,True)>cv2.arcLength(rightBox,True)):
							rightBox = approx
							slopeRight = slope
					else:
						RIGHT_LINE = True
						rightBox = approx 
						slopeRight = slope

		steering_angle = 0
		reward = 0
		NO_LINE = False

		if(LEFT_LINE and RIGHT_LINE):
			np_left = np.squeeze(leftBox)
			np_right = np.squeeze(rightBox)
			
			right_pX = ( min(np_right[:,0]) + (max(np_right[:,0])) )/2
			left_pX = ( min(np_left[:,0]) + (max(np_left[:,0])) )/2

			pointX = int((left_pX+right_pX)/2)-10
			#pointX = int((np.average(test[:,0])+np.average(test1[:,0]))/2)
			left_pointY = int((min(np_left[:,1])+max(np_left[:,1]))/2)
			right_pointY = int((min(np_right[:,1])+max(np_right[:,1]))/2)

			#pointY = (left_pointY+right_pointY)/2#min(pointY1, pointY2)
			#rightWidth = (244-max(test1[:,0]))
			#pointX-= rightWidth
			#leftWidth = (min(test[:,0]))
			#pointX+=leftWidth

			if(im_out == "contours"):
				cv2.drawContours(img2, [leftBox], -1, (255,255,255), 2)
				cv2.drawContours(img2, [rightBox], -1, (255,255,255), 2)
		else:

		# Correct misclassified line / \
			# Should be positive
			if(LEFT_LINE): # /
				if(slopeLeft < -1):
					rightBox = leftBox
					LEFT_LINE = False
					RIGHT_BOX = True
			# Should be negative
			elif(RIGHT_LINE): # \
				if(slopeRight > 1):
					leftBox = rightBox
					RIGHT_BOX = False
					LEFT_BOX = True

			# Left Line /
			if(LEFT_LINE):
				np_left = np.squeeze(leftBox)
				pointX = max(np_left[:,0])+50
				#pointY = int((min(test[:,1])+max(test[:,1]))/2)
				if(im_out == "contours"):
					cv2.drawContours(img2, [leftBox], -1, (255,255,255), 2)

			# Right line \
			elif(RIGHT_LINE):
				np_right = np.squeeze(rightBox)
				pointX = min(np_right[:,0])-50
				#pointY = int((min(test[:,1])+max(test[:,1]))/2)
				if(im_out == "contours"):
					cv2.drawContours(img2, [rightBox], -1, (255,255,255), 2)
			else:
				NO_LINE = True
				pointX = 0

		# bad
		try:
			pointY = 122
			centerX = int(244/2)
			centerY = int(244)

			angle_to_mid_radian = math.atan((pointX-122)/(244-pointY))
			angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
			steering_angle = (angle_to_mid_deg)+90
						
			# Heading stabilizing-- change later
			if(pointX > 244):
				pointX = 180
			elif(pointX <= 0):
				pointX = 60

			if(im_out is not None):
				# TARGET DOT
				cv2.line(img2,(centerX,centerY),(pointX,pointY),(255,255,255),3)
				cv2.circle(img2, (int(pointX), int(122)), 4, (255,255,255),  -1)

			#CTE
			reward = 1-(math.sqrt( ((pointY-230)**2)+((pointX-122)**2) )/150)
			
			self.angle = steering_angle
			self.line = np.array([pointX])
			self.reward = reward
		except:
			print("Err")
			steering_angle = -1
			reward = 0
			
		# PD Calculations
        error = (self.angle-90) # -90 0 90 ERR
        P_T = error * self.P
        D_T = ((error-self.error_old)/self.CONST_TIME)*self.D
        cv_action = int(np.clip(((P_T-D_T)+90), 2, 178))
        self.error_old = error
		self.preview_img = img2
		
		return cv_action