import glob
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def get_calibration_points(image_dir, nx = 9, ny = 6):
	# load calibration images file names
	images = glob.glob(image_dir + 'calibration*.jpg')
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((ny*nx, 3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
	# arrays to store object points and image points from all the images.
	objpoints = [] 
	imgpoints = [] 
	for fname in images:
		img = cv2.imread(fname)
		# Resize the image into reasonable size
		img = cv2.resize(img, (720, 405))
		# convert BGR image to Gray image
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		if ret == True:
			corners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
			objpoints.append(objp)
			imgpoints.append(corners)
	return imgpoints, objpoints


def undistort_image(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def get_unwarp_matrix(src, dst):
	M = cv2.getPerspectiveTransform(src, dst)
	return M


def unwarp_image(img, M):
	img_size = (img.shape[1], img.shape[0])
	warped = cv2.warpPerspective(img, M, img_size)
	return warped


def abs_sobel_thresh(gray_img, orient = 'x', sobel_kernel = 3, thresh = (30, 100)):
    # 1) Take the derivative in x or y given the orient value
    if orient == 'x':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 2) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 4) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(gray_img, sobel_kernel = 3, mag_thresh = (30, 100)):
    # 1) Take the derivative in x and y 
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 2) Take the absolute value of the derivative or gradient
    abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 4) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output
    

def dir_threshold(gray_img, sobel_kernel = 3, thresh = (0, np.pi/2)):
    # 1) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 2) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 3) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 4) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(abs_sobelx)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output



if __name__ == '__main__':

	cal_images_dir = 'camera_cal/'
	# calculate iamge points and object points needed for camera calibration
	imgpoints, objpoints = get_calibration_points(image_dir = cal_images_dir)
	# Load test image
	img = cv2.imread('image1.jpg')
	# Image processing pipeline:
	# 1) Convert img from BGR to RGB space
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# 2) Resize the img into 720 x 405
	rgb_img = cv2.resize(rgb_img, (720, 405))
	# 3) Convert the img from RGB to Gray scale
	gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
	# Defien kernel size
	ksize = 3
	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(gray_img, orient = 'x', sobel_kernel = ksize, thresh=(20, 100))
	grady = abs_sobel_thresh(gray_img, orient = 'y', sobel_kernel = ksize, thresh=(20, 100))
	mag_binary = mag_thresh(gray_img, sobel_kernel = ksize, mag_thresh = (30, 100))
	dir_binary = dir_threshold(gray_img, sobel_kernel = ksize, thresh = (0.7, 1.3))

	combined_binary = np.zeros_like(dir_binary)
	combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

	# grad_binary_img = abs_sobel_thresh(gray_img, orient='x', thresh_min = 20, thresh_max = 100)
	# mag_binary_img = mag_thresh(gray_img, sobel_kernel = 3, mag_thresh = (30, 100))
	# dir_binary_img = dir_threshold(gray_img, sobel_kernel = 3, thresh = (0.7, 1.3))

	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
	f.tight_layout()
	ax1.imshow(rgb_img)
	ax1.set_title('Original Image', fontsize = 10)
	ax2.imshow(combined_binary, cmap = 'gray')
	ax2.set_title('Combined Gradient', fontsize = 10)
	plt.subplots_adjust(left = 0., right = 1, top = 0.9, bottom = 0.)
	plt.show()

	# # get undistorted image
	# undist_img = undistort_image(image, objpoints, imgpoints)
	# # define source points and destination points for perspective transform
	# apex, apey = 360, 258
	# offset_far = 50
	# offset_near = 10
	# src_points = np.float32([[int(apex - offset_far), apey],
	# 	[int(apex + offset_far), apey],
	# 	[int(0 + offset_near), 390],
	# 	[int(720-offset_near), 390]])
	# dst_points = np.float32([[0, 0], [720, 0], [0, 405], [720, 405]])

	# # Get perspective_matrix from src_points and dst_points
	# perspective_matrix = get_unwarp_matrix(src_points, dst_points)
	# # Get unwarp_image from undist_img using perspective_matrix
	# unwarp_img = unwarp_image(undist_img, perspective_matrix)

	# f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))
	# ax1.imshow(image)
	# ax1.set_title('Original Image', fontsize = 10)
	# ax2.imshow(undist_img)
	# ax2.set_title('Undistort Image', fontsize = 10)
	# ax3.imshow(unwarp_img)
	# ax3.set_title('Unwarp Image', fontsize = 10)
	# f.subplots_adjust(hspace = 0.3)
	# plt.show()

