import cv2
import numpy as np
import math

cam = cv2.VideoCapture(1)
lower_black = [0, 0, 0]
upper_black = [180, 255, 50]
def detect_all_squares(image):
	squares = []
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)
	test = cv2.bitwise_and(image, image, mask = mask)
	cv2.imshow('test', test)
	gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 127, 255, 0)
	cv2.imshow('mask', mask)
	contours,  h = cv2.findContours(mask, 1, 2)
	for cnt in contours:
		epsilon = 0.15 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if area > 40:
			if len(approx) == 4:
				cv2.drawContours(image, [cnt], 0, (255, 0, 0), 1)
				squares = squares + [cnt]
	cv2.imshow('img', image)
	return squares

def sorted_areas(contours):
	areas = []
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
	return required_contour

def angle(x1, x2, y1, y2):
	if x1 == x2:
		return 90
	return math.atan(float(y1 - y1) / (x1 - x2))

def positioned(pt1, pt2, pt3):
	angle1 = angle(pt1[0], pt2[0], pt1[1], pt2[1])
	angle2 = angle(pt1[0], pt3[0], pt1[1], pt3[1])
	print angle1, angle2
	if angle1 - angle2 != 90:
		return False
	return True

def main():
	positioning_squares = []
	while True:
		cv2.waitKey(30)
		ret, image = cam.read()
		cv2.imshow('blue', image[:, :, 0])
		cv2.imshow('green', image[:, :, 1])
		cv2.imshow('red', image[:, :, 2])
		image = cv2.GaussianBlur(image, (3,3), 0)
		if not ret:
			image = cv2.imread("qr.png")
		square_contours = detect_all_squares(image)
		positioning_squares = sorted_areas(square_contours)
		for coor in positioning_squares :
			cv2.drawContours(image, [coor], 0, (0, 0, 255), 1)
		cv2.imshow('image', image);
		print "trying to get the corner squares......"
		if (len(positioning_squares) > 2):
			upper_left_corner = positioning_squares[0][0][0]
			bottom_left_corner = positioning_squares[2][1][0]
			upper_right_corner = positioning_squares[1][3][0]
		else:
			continue
		if not positioned(upper_left_corner, bottom_left_corner, upper_right_corner):
			continue
		cv2.imwrite('snap_img.png', image)
		break
		k = cv2.waitKey(30)
		if k == 27:
			break
	cv2.destroyAllWindows()
	cam.release()


if __name__ == "__main__":
	main()
