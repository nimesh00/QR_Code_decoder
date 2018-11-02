#! /usr/bin/python

import cv2
import numpy as np
import math

lower_black = [0, 0, 0]
upper_black = [0, 0, 10]
def detect_all_squares(image):
	squares = []
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)
	cv2.waitKey(30)
	img, contours,  h = cv2.findContours(mask, 1, 2)
	for cnt in contours:
		epsilon = 0.15 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if area > 400:
			if len(approx) == 4:
				squares = squares + [cnt]
	return squares
	
def detect_all_squares_again(image):
	squares = []
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)
	cv2.waitKey(30)
	img, contours,  h = cv2.findContours(mask, 1, 2)
	for cnt in contours:
		epsilon = 0.3 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if area > 400:
			if len(approx) == 4:
				squares = squares + [cnt]
	return squares

def sorted_areas(contours):
	areas = []
	print "total contours: ", len(contours)
	required_areas = []
	required_contour = []
	no_of_contours = 0
	for cnt in contours:
		no_of_contours += 1
		if no_of_contours == 1:
			largest_area = cv2.contourArea(cnt)
			requried_contour = required_contour + [cnt]
		area = cv2.contourArea(cnt)
		if area >= largest_area:
			largest_area = area
			required_areas = [area] + required_areas
			required_contour = [cnt] + required_contour

		else:
			no_of_contour_here = 0
			for cnt1 in required_contour:
				no_of_contour_here += 1
				if no_of_contour_here == len(required_contour):
					required_contour = required_contour + [cnt]
					break
				elif area > cv2.contourArea(cnt1):
					required_contour[no_of_contour_here:] = [cnt] + required_contour[no_of_contour_here:]
					break
				elif area <= cv2.contourArea(cnt1):
					required_contour[: no_of_contour_here + 1] = required_contour[: no_of_contour_here + 1] + [cnt]
					break
		required_contour = required_contour[:9]
		#print len(required_contour)
	
	return required_contour

def make_grid(pos_sq, imag):
	upper_left = pos_sq[0][0][0]
	bottom_left = pos_sq[2][1][0]
	upper_right = pos_sq[1][3][0]
	wx = (abs(upper_right[0] - upper_left[0]) / 21)
	wy = (abs(bottom_left[1] - upper_left[1]) / 21)
	x_iter = upper_left[0]
	y_iter = upper_left[1]
	print "wx: ", wx, "wy: ", wy
	while x_iter < upper_right[0]:
		cv2.line(imag, (x_iter, y_iter), (x_iter, bottom_left[1]), (0, 255, 255), 1)
		x_iter += wx
	x_iter = upper_left[0]
	while y_iter < bottom_left[1]:
		cv2.line(imag, (x_iter, y_iter), (upper_right[0], y_iter), (0, 255, 255), 1)
		y_iter += wy
	cv2.imshow('grided', imag)
	return imag

def mark_data_squares(image):
	squares = []
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)
	img, contours,  h = cv2.findContours(mask, 1, 2)
	for cnt in contours:
		epsilon = 0.005 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if len(approx) == 4:
			cv2.drawContours(image, [cnt], 0, (0, 0, 255), 1)
			squares = squares + [cnt]
	return squares
'''
def generate_array(img, ul, ur, bl):
	array = [[0 for i in range(21)] for j in range(21)]
	qr = img[ul[1]:bl[1], ul[0]:ur[0]]
	qr = cv2.resize(qr, (21, 21), interpolation = cv2.INTER_AREA)
	blue_channel = qr[:, :, 0]
	for i in range(21):
		for j in range(21):
			if blue_channel[i, j] < 127:
				array[i][j] = 1
			else:
				array[i][j] = 0
	for k in range(21):
		print array[k]
	cv2.imshow('clipped', qr)
'''
def distance(x1, y1, x2, y2):
	return int(np.sqrt((x1 - x2)**2 + (y1 - y2)**2))

def rotate(img, angle):
	height, width = img.shape[:2]
	M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
	rot = cv2.warpAffine(img, M, (height, width))
	return rot

def angle(xp,yp,xb,yb):
    dx=float(xp-xb)
    dy=float(yp-yb)
    if dx == 0:
        return 90
    global mtan
    if(dx > 0 and dy > 0):
        mtan=math.degrees(math.atan(float(dy/dx)))
    elif(dy>0 and dx<0):
        mtan=180 + math.degrees(math.atan(float(dy/dx)))
    elif(dy <0 and dx<0):
        mtan=180+math.degrees(math.atan(float(dy/dx)))
    else:
        mtan=360+math.degrees(math.atan(float(dy/dx)))
    #print mtan
    return mtan


def get_small_area_image(img, p1, p2, p0):
	''' NOT COMPLETE - MORE CASES TO BE ADDED FOR CORRECT ORIENTATION IN EVERY ORIENTETION'''
	x_coor = []
	y_coor = []
	max_dis_bw = []
	points = [p0, p1, p2]
	for i in range(3):
		for cor in points[i]:
			x_coor.append(cor[0][0])
			y_coor.append(cor[0][1])
	xmin = np.amin(x_coor)
	xmax = np.amax(x_coor)
	ymin = np.amin(y_coor)
	ymax = np.amax(y_coor)
	distance1 = distance(p0[0][0][0], p0[0][0][1], p1[0][0][0], p1[0][0][1])
	max_dis_bw = [p0, p1]
	distance2 = distance(p0[0][0][0], p0[0][0][1], p2[0][0][0], p2[0][0][1])
	distance3 = distance(p1[0][0][0], p1[0][0][1], p2[0][0][0], p2[0][0][1])
	if distance2 > distance1 and distance1 > distance3:
		max_dis_bw = [p0, p2]
	if distance3 > distance1 and distance3 > distance2:
		max_dis_bw = [p1, p2]
	for p in points:
		if p in max_dis_bw:
			continue
		else:
			rotation_angle = angle(p[0][0][0], p[0][0][1], p1[0][0][0], p1[0][0][1])
	y_correction = distance(xmin, ymin, xmax, ymax) - (ymax - ymin)
	x_correction = distance(xmin, ymin, xmax, ymax) - (xmax - xmin)
	small_img = img[ymin - y_correction:ymax + y_correction, xmin - x_correction:xmax + x_correction]
	
	small_img = rotate(small_img, -1 * rotation_angle)
	
	#cv2.imshow('small_one', small_img)
	return small_img

def normalised(qr):
	height, width = qr.shape[:2]
	block_height = height / 21
	block_width = width / 21
	block_total = 0
	array = [[0 for i in range(21)] for j in range(21)]
	for i in range(21):
		for j in range(21):
			block_total = 0
			for k in range(block_height):
				for l in range(block_width):
					block_total += qr[block_height * i + k][block_width * j + l]
			array[i][j] = block_total / (block_height * block_width)
	'''
	print "normalised values"
	for i in range(21):
		print array[i]
	'''
	return array

def smaller(h1, h2):
	if h1 < h2:
		return h1
	else:
		return h2

def greater(h1, h2):
	if h1 > h2:
		return h1
	else:
		return h2
		
		
def get_corner_points(p0, p1, p2):
	x_coor = []
	y_coor = []
	points = [p0, p1, p2]
	for i in range(3):
		for cor in points[i]:
			x_coor.append(cor[0][0])
			y_coor.append(cor[0][1])
	xmin = np.amin(x_coor)
	xmax = np.amax(x_coor)
	ymin = np.amin(y_coor)
	ymax = np.amax(y_coor)
	ul = [p0[0][0][0], p0[0][0][1]]
	ur = [p0[0][0][0], p0[0][0][1]]
	bl = [p0[0][0][0], p0[0][0][1]]
	#print "x_coor: ", x_coor
	#print 'x y max min', xmax, xmin, ymax, ymin
	for i in range(3):
		for coor in points[i]:
			if distance(xmin, ymin, coor[0][0], coor[0][1]) < distance(xmin, ymin, ul[0], ul[1]):
				ul = [coor[0][0], coor[0][1]]
			if distance(xmax, ymin, coor[0][0], coor[0][1]) < distance(xmax, ymin, ur[0], ur[1]):
				ur = [coor[0][0], coor[0][1]]
			if distance(xmin, ymax, coor[0][0], coor[0][1]) < distance(xmin, ymax, bl[0], bl[1]):
				bl = [coor[0][0], coor[0][1]]
	br = [ur[0], bl[1]]
	
	return ul, ur, bl, br

		
def get_extremites(points):
	x_coor = []
	y_coor = []
	sq1 = sq2 = sq3 = points[0]
	for i in range(len(points)):
		for cor in points[i]:
			x_coor.append(cor[0][0])
			y_coor.append(cor[0][1])
	xmin = np.amin(x_coor)
	xmax = np.amax(x_coor)
	ymin = np.amin(y_coor)
	ymax = np.amax(y_coor)
	ul = [points[0][0][0][0], points[0][0][0][1]]
	ur = [points[0][0][0][0], points[0][0][0][1]]
	bl = [points[0][0][0][0], points[0][0][0][1]]
	square_array = []
	#print "x_coor: ", x_coor
	#print 'x y max min', xmax, xmin, ymax, ymin
	'''
	for i in range(len(points)):
		for coor in points[i]:
			if distance(xmin, ymin, coor[0][0], coor[0][1]) < distance(xmin, ymin, ul[0], ul[1]):
				ul = [coor[0][0], coor[0][1]]
				sq1 = coor
			if distance(xmax, ymin, coor[0][0], coor[0][1]) < distance(xmax, ymin, ur[0], ur[1]):
				ur = [coor[0][0], coor[0][1]]
				sq2 = coor
			if distance(xmin, ymax, coor[0][0], coor[0][1]) < distance(xmin, ymax, bl[0], bl[1]):
				bl = [coor[0][0], coor[0][1]]
				sq3 = coor
	'''
	for coor in points:
		for i in range(len(coor)):
			if distance(xmin, ymin, coor[i][0][0], coor[i][0][1]) <= distance(xmin, ymin, ul[0], ul[1]):
				ul = [coor[i][0][0], coor[i][0][1]]
				sq1 = coor
			if distance(xmax, ymin, coor[i][0][0], coor[i][0][1]) <= distance(xmax, ymin, ur[0], ur[1]):
				ur = [coor[i][0][0], coor[i][0][1]]
				sq2 = coor
			if distance(xmin, ymax, coor[i][0][0], coor[i][0][1]) <= distance(xmin, ymax, bl[0], bl[1]):
				bl = [coor[i][0][0], coor[i][0][1]]
				sq3 = coor
	
	square_array = square_array + [sq1] + [sq2] + [sq3]
	'''
	for i in range(len(square_array)):
		print square_array[i][0]
	'''
	return square_array

def positionGray(img):
	'''
	if len(img[0,0]) != 1:
		img = cv2.GaussianBlur(img, (1, 1), 0)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	'''
	ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
	ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
	
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	positioning_squares = []
	square_contours = detect_all_squares_again(img)
	'''
	for cnt in square_contours:
		#print cnt[0]
	'''
	#print "square contours:", square_contours
	positioning_squares = sorted_areas(square_contours)
	positioning_squares = get_extremites(positioning_squares)
	for cnt in positioning_squares:
		cv2.drawContours(img, [cnt], 0, (0, 255, 0), 1)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	#print "positioning squares 1:", positioning_squares[0]
	ul, ur, bl, br = get_corner_points(positioning_squares[1], positioning_squares[2], positioning_squares[0])
	return ul, ur, bl, br

def position(img):
	img = cv2.GaussianBlur(img, (1, 1), 0)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
	ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)	
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	positioning_squares = []
	square_contours = detect_all_squares(img)
	'''
	for cnt in square_contours:
		#print cnt[0]
	'''
	#print "square contours:", square_contours
	positioning_squares = sorted_areas(square_contours)
	positioning_squares = get_extremites(positioning_squares)
	for cnt in positioning_squares:
		cv2.drawContours(img, [cnt], 0, (0, 255, 0), 1)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	#print "positioning squares 1:", positioning_squares[0]
	ul, ur, bl, br = get_corner_points(positioning_squares[1], positioning_squares[2], positioning_squares[0])
	return ul, ur, bl, br

def vertical_shear_normalise(qr, ul, ur, bl, br):
	print "given four points: ", ul, ur, bl, br
	height, width = qr.shape[:2]
	#print "old width : ", width
	width = int(float(width / (ur[0] - ul[0])) * 210)
	#print "new width : ", width
	qr2 = cv2.resize(qr, (width, height), interpolation = cv2.INTER_AREA)
	ul, ur, bl, br = positionGray(qr2)
	cv2.imshow('resized', qr2)
	cv2.waitKey(0)
	#block_width = (ur[0] - ul[0]) / 21
	block_width = 10
	block_height = 10
	block_total = 0
	array = [[0 for i in range(21)] for j in range(21)]
	for j in range(21):
		X = ul[0] + j * block_width
		#print "x value: ", X
		#print "slope: ",(float((ur[1] - ul[1])) / (ur[0] - ul[0]))
		ya = ((float(ur[1] - ul[1]) / (ur[0] - ul[0])) * X) + (ul[1] * ur[0] - ur[1] * ul[0]) / (ur[0] - ul[0])
		yb = ((float(br[1] - bl[1]) / (br[0] - bl[0])) * X) + (bl[1] * br[0] - br[1] * bl[0]) / (br[0] - bl[0])
		#ya = (((ur[1] - ul[1]) * X) + (ul[1] * ur[0] - ur[1] * ul[0])) / (ur[0] - ul[0])
		#print "y coordinates(ya yb): ", ya, yb
		
		#block_height = int((yb - ya) / 21)
		height = int(float(height / (yb - ya)) * 210)
		qr2 = cv2.resize(qr, (width, height), interpolation = cv2.INTER_AREA)
		cv2.imshow('resized', qr2)
		#cv2.waitKey(0)
		ul, ur, bl, br = positionGray(qr2)
		ya = ((float(ur[1] - ul[1]) / (ur[0] - ul[0])) * X) + (ul[1] * ur[0] - ur[1] * ul[0]) / (ur[0] - ul[0])
		yb = ((float(br[1] - bl[1]) / (br[0] - bl[0])) * X) + (bl[1] * br[0] - br[1] * bl[0]) / (br[0] - bl[0])
		#print "width, height", block_width, block_height
		for i in range(21):
			#block_height = (((h_diff / 21) * j) + (bl[1] - smaller(ul[1], ur[1]))) / 21
			y_curr = int(ya)
			block_total = 0
			for k in range(block_height):
				for l in range(block_width):
					block_total += qr2[y_curr + block_height * i + k][X + l]
			array[i][j] = block_total / (block_height * block_width)
			
	return array

def generate_array(img, ul, ur, bl):
	br = [ur[0], bl[1]]
	array = [[0 for i in range(21)] for j in range(21)]
	#qr = img[ul[1]:bl[1], ul[0]:ur[0]]
	qr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#qr = cv2.resize(qr, (210, 210), interpolation = cv2.INTER_AREA)
	#averaged_array = normalised(qr)
	averaged_array = vertical_shear_normalise(qr, ul, ur, bl, br)
	for i in range(21):
		for j in range(21):
			if averaged_array[i][j] < 127:
				if ((i < 7 or i > 13) and j < 7)  or ((j < 7 or j > 13) and i < 7):
					array[i][j] = 2
				else:
					array[i][j] = 1
			else:
				array[i][j] = 0
	for k in range(21):
		print array[k]
	cv2.imshow('clipped', qr)
	return array

'''
def get_corner_points(p0, p1, p2):
	x_coor = []
	y_coor = []
	points = [p0, p1, p2]
	for i in range(3):
		for cor in points[i]:
			x_coor.append(cor[0][0])
			y_coor.append(cor[0][1])
	xmin = np.amin(x_coor)
	xmax = np.amax(x_coor)
	ymin = np.amin(y_coor)
	ymax = np.amax(y_coor)
	ul = [p0[0][0][0], p0[0][0][1]]
	ur = [p0[0][0][0], p0[0][0][1]]
	bl = [p0[0][0][0], p0[0][0][1]]
	print 'x y max min', xmax, xmin, ymax, ymin
	for i in range(3):
		for coor in points[i]:
			if distance(xmin, ymin, coor[0][0], coor[0][1]) < distance(xmin, ymin, ul[0], ul[1]):
				ul = [coor[0][0], coor[0][1]]
			if distance(xmax, ymin, coor[0][0], coor[0][1]) < distance(xmax, ymin, ur[0], ur[1]):
				ur = [coor[0][0], coor[0][1]]
			if distance(xmin, ymax, coor[0][0], coor[0][1]) < distance(xmin, ymax, bl[0], bl[1]):
				bl = [coor[0][0], coor[0][1]]
	br = [ur[0], bl[1]]
	
	return ul, ur, bl, br
'''

def get_mask_array(mask_number):
	array = [[0 for i in range(21)] for j in range(21)]
	for i in range(21):
		for j in range(21):
			if mask_number == 0:
				if ((i * j) % 2 + (i * j) %3) == 0:
					array[i][j] = 1
				else:
					array[i][j] = 0
			elif mask_number == 1:
				if (i / 2 + j / 3) % 2 == 0:
					array[i][j] = 1
				else:
					array[i][j] = 0
			elif mask_number == 2:
				if ((i * j) % 3 + i + j) % 2 == 0:
					array[i][j] = 1
				else:
					array[i][j] = 0
			elif mask_number == 3:
				if ((i * j) % 3 + i * j) % 2 == 0:
					array[i][j] = 1
				else:
					array[i][j] = 0
			elif mask_number == 4:
				if i % 2 == 0:
					array[i][j] = 1
				else:
					array[i][j] = 0
			elif mask_number == 5:
				if (i + j) % 2 == 0:
					array[i][j] = 1
				else:
					array[i][j] = 0
			elif mask_number == 6:
				if (i + j) % 3 == 0:
					array[i][j] = 1
				else:
					array[i][j] = 0
			elif mask_number == 7:
				if j % 3 == 0:
					array[i][j] = 1
				else:
					array[i][j] = 0
	'''
	print "Mask Pattern:"
	for k in range(21):
		print array[k]
	'''
	return array

def unmask(array):
	number = 0
	number = 4 * array[8][2] + 2 * array[8][3] + array[8][4]
	mask_pattern_array = get_mask_array(number)
	
	'''---TODO---SPECIAL AREAS NOT CONSIDERED RIGHT NOW. DO THAT FAST'''
	for i in range(21):
		for j in range(21):
			if (mask_pattern_array[i][j] == 1):
				if (array[i][j] == 1):
					array[i][j] = 0
				elif array[i][j] == 0:
					array[i][j] = 1
	return array

def decode(array):
	format_no = 8 * array[20][20] + 4 * array[20][19] + 2 * array[19][20] + array[19][19]
	if format_no == 0:
		print "kuch nahi pata"
	elif format_no == 1:
		print "Numeric encoding"
		data_bit_length = 10
		data_byte = [0 for i in range(data_bit_length)]
		msg_byte = [0 for i in range(data_bit_length)]
		required_data = 0
		msg_length = 0
		'''
		for j in range(18, 14, -1):
			if i == 20:
				i = 19
			elif i == 19:
				i = 20
			y_pointer = 18 - j
			x_pointer = 20 - i
			msg_byte[index] = array[i][j]
		'''
		y_pointer = 37
		x_curr = 20
		column_no = 1
		for i in range(data_bit_length):
			y_curr = int((37 - i) / 2)
			msg_byte[i] = array[y_curr][x_curr]
			msg_length += msg_byte[i] * (2 ** (9 - i))
			if column_no == 1:
				if x_curr == 19 - 2 * (column_no - 1):
					x_curr = 20 - 2 * (column_no - 1)
				elif x_curr == 20 - 2 * (column_no - 1):
					x_curr = 19 - 2 * (column_no - 1)
			
		y_pointer = 2 * (y_curr - 1) + 1

		for i in range(data_bit_length):
			y_curr = int((y_pointer - i) / 2)
			data_byte[i] = array[y_curr][x_curr]
			if column_no == 1:
				if x_curr == 19 - 2 * (column_no - 1):
					x_curr = 20 - 2 * (column_no - 1)
				elif x_curr == 20 - 2 * (column_no - 1):
					x_curr = 19 - 2 * (column_no - 1)
			
		print "data byte:",data_byte
		k = data_bit_length - 1
		while (data_byte[k] == 0) and ((data_bit_length - k) < 7):
			k -= 1
		index = k
		while k >= 0:
			required_data += data_byte[k] * (2 ** (index - k))
			k -= 1
		return required_data



def main():
	positioning_squares = []
	font = cv2.FONT_HERSHEY_SIMPLEX
	image = cv2.imread("snap_img.png")
	
	ret, image[:, :, :] = cv2.threshold(image[:, :, :], 110, 255, cv2.THRESH_BINARY)
	#image = cv2.GaussianBlur(image, (5,5), 0)
	image2 = cv2.imread("snap_img.png")
	cv2.imshow('thresholding', image)
	square_contours = detect_all_squares(image)
	positioning_squares = sorted_areas(square_contours)
	#gridImage = make_grid(positioning_squares, image2)
	#data_squares = mark_data_squares(gridImage)
	#print "contours returned: ", len(positioning_squares)
	i = 0
	#cv2.drawContours(image2, [positioning_squares[3]], 0, (0, 255, 0), 1)
	#cv2.imshow('image', image)
	for coor in positioning_squares:
		#print "coordinate",i, "area", cv2.contourArea(coor),": ", coor
		i += 1
		cv2.drawContours(image2, [coor], 0, (0, 0, 255), 1)
	cv2.waitKey(0)
	
	small_image = get_small_area_image(image2, positioning_squares[1], positioning_squares[2], positioning_squares[0])
	small_image = cv2.GaussianBlur(small_image, (3, 3), 0)
	small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
	ret, small_image = cv2.threshold(small_image, 100, 255, cv2.THRESH_TOZERO)
	small_image = cv2.GaussianBlur(small_image, (3, 3), 0)
	ret, small_image = cv2.threshold(small_image, 100, 255, cv2.THRESH_BINARY)
	small_image = cv2.cvtColor(small_image, cv2.COLOR_GRAY2BGR)
	cv2.imshow('small_with_corner', small_image)
	cv2.waitKey(0)
	cv2.imwrite('small_qr.png', small_image)
	small2 = cv2.imread('small_qr.png')
	#print 'blue channel: ', small_image[1, 1, 0]
	#small_image[:, :, :1] = [0, 0, 0]
	#ret, small_image[:, :, :] = cv2.threshold(small_image[:, :, :], 127, 255, cv2.THRESH_BINARY)
	'''
	height, width = small_image.shape[:2]
	print height, width
	for i in range(height):
		for j in range(width):
			if small_image[i, j, 0] + small_image[i, j, 1] + small_image[i, j, 2] > 200:
				for k in range(3):
					small_image[i, j, k] = 255
			else:
				for k in range(3):
					small_image[i, j, k] = 0
	'''
	cv2.imshow('small_thresh', small_image)
	#small_image = cv2.GaussianBlur(small_image, (5,5), 0)
	square_contours = detect_all_squares(small_image)
	positioning_squares = sorted_areas(square_contours)
	#cv2.drawContours(small2, [positioning_squares[3]], 0, (0, 255, 0), 1)
	for coor in positioning_squares:
		#print "coordinate",i, "area", cv2.contourArea(coor),": ", coor
		i += 1
		cv2.drawContours(small2, [coor], 0, (0, 0, 255), 1)
	'''
	upper_left_corner = positioning_squares[1][0][0]
	bottom_left_corner = positioning_squares[2][1][0]
	upper_right_corner = positioning_squares[0][3][0]
	'''
	#upper_left_corner, upper_right_corner, bottom_left_corner, bottom_right_corner = get_corner_points(positioning_squares[1], positioning_squares[2], positioning_squares[0])
	upper_left_corner, upper_right_corner, bottom_left_corner, bottom_right_corner = position(small_image)
	qr_array = generate_array(cv2.imread('small_qr.png'), upper_left_corner, upper_right_corner, bottom_left_corner)
	
	unmasked_qr = unmask(qr_array)
	message = decode(unmasked_qr)
	print "Final number", message
	
	cv2.circle(small2, (upper_left_corner[0], upper_left_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText(small2, "upper_left_corner", (upper_left_corner[0], upper_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.circle(small2, (bottom_left_corner[0], bottom_left_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText(small2, "bottom_left_corner", (bottom_left_corner[0], bottom_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.circle(small2, (upper_right_corner[0], upper_right_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText(small2, "upper_right_corner", (upper_right_corner[0], upper_right_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow('final contours', small2)
	
	
	image_to_show = cv2.imread('snap_img.png')
	height, width = small2.shape[:2]
	for i in range(height):
		for j in range(width):
			image_to_show[i, j] = small2[i, j]
	cv2.imshow('original with final contours', image_to_show)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
