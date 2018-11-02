#! /usr/bin/python

import cv2
import numpy as np

lower_black = [0, 0, 0]
upper_black = [0, 0, 10]
def detect_all_squares(image):
	squares = []
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)
	cv2.waitKey(30)
	img, contours,  h = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_SIMPLE)
	print 'contours: ',contours
	for cnt in contours:
		epsilon = 0.15 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if area > 400:
			if len(approx) == 4:
				squares = squares + [cnt]
	return squares

def sorted_areas(contours):
	areas = []
	#print "total contours: ", len(contours)
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
'''
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
				if ((i < 7 or i > 13) and j < 7)  or ((j < 7 or j > 13) and i < 7):
					array[i][j] = 2
				else:
					array[i][j] = 1
			else:
				array[i][j] = 0
	'''
	for k in range(21):
		print array[k]
	'''
	cv2.imshow('clipped', qr)
	return array

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
	
	# ---TODO---SPECIAL AREAS NOT CONSIDERED RIGHT NOW. DO THAT FAST
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
	image = cv2.imread("qr_6.png")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	cv2.imwrite('qr.png', image)
	image2 = cv2.imread("qr.png")
	square_contours = detect_all_squares(image)
	positioning_squares = sorted_areas(square_contours)
	#print "contours returned: ", len(positioning_squares)
	i = 0
	cv2.drawContours(image, [positioning_squares[3]], 0, (0, 255, 0), 1)
	cv2.imshow('image', image)
	for coor in positioning_squares:
		#print "coordinate",i, "area", cv2.contourArea(coor),": ", coor
		i += 1
		cv2.drawContours(image, [coor], 0, (0, 0, 255), 1)
	
	
	upper_left_corner = positioning_squares[0][0][0]
	bottom_left_corner = positioning_squares[2][1][0]
	upper_right_corner = positioning_squares[1][3][0]
	qr_array = generate_array(cv2.imread('qr.png'), upper_left_corner, upper_right_corner, bottom_left_corner)
	unmasked_qr = unmask(qr_array)
	
	message = decode(unmasked_qr)
	print "Final number", message
	
	print 'unamsked qr data:'
	for i in range(21):
		print unmasked_qr[i]
		
	cv2.circle(image, (upper_left_corner[0], upper_left_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText( image, "upper_left_corner", (upper_left_corner[0], upper_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.circle(image, (bottom_left_corner[0], bottom_left_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText( image, "bottom_left_corner", (bottom_left_corner[0], bottom_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.circle(image, (upper_right_corner[0], upper_right_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText( image, "upper_right_corner", (upper_right_corner[0], upper_right_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow('image', image)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
