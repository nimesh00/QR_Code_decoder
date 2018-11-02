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
	img, contours,  h = cv2.findContours(mask, 1, 2)
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
		print len(required_contour)
	
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

def main():
	positioning_squares = []
	font = cv2.FONT_HERSHEY_SIMPLEX
	image = cv2.imread("qr_dev.png")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	cv2.imwrite('qr.png', image)
	image2 = cv2.imread("qr.png")
	square_contours = detect_all_squares(image)
	positioning_squares = sorted_areas(square_contours)
	gridImage = make_grid(positioning_squares, image2)
	data_squares = mark_data_squares(gridImage)
	print "contours returned: ", len(positioning_squares)
	i = 0
	cv2.drawContours(image, [positioning_squares[3]], 0, (0, 255, 0), 1)
	cv2.imshow('image', image)
	for coor in positioning_squares:
		print "coordinate",i, "area", cv2.contourArea(coor),": ", coor
		i += 1
		cv2.drawContours(image, [coor], 0, (0, 0, 255), 1)
	
	
	upper_left_corner = positioning_squares[0][0][0]
	bottom_left_corner = positioning_squares[2][1][0]
	upper_right_corner = positioning_squares[1][3][0]
	qr_array = generate_array(cv2.imread('qr.png'), upper_left_corner, upper_right_corner, bottom_left_corner)
	cv2.circle(image, (upper_left_corner[0], upper_left_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText( image, "upper_left_corner", (upper_left_corner[0], upper_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.circle(image, (bottom_left_corner[0], bottom_left_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText( image, "bottom_left_corner", (bottom_left_corner[0], bottom_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.circle(image, (upper_right_corner[0], upper_right_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText( image, "upper_right_corner", (upper_right_corner[0], upper_right_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow('image', image)
	
	cv2.imshow('grided', gridImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
