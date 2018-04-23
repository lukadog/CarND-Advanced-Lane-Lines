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
	Minv = cv2.getPerspectiveTransform(dst, src)
	return M, Minv


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


def color_threshold(img, channel = 2, thresh = (0, 255)):
	color_channel = img[:, :, channel]
	binary_output = np.zeros_like(color_channel)
	binary_output[(color_channel >= thresh[0]) & (color_channel <= thresh[1])] = 1
	return binary_output


def process_image_pipeline(img, objpoints, imgpoints, perspective_matrix):
	# 1) Resize image
	rgb_img = cv2.resize(img, (720, 405))
	# 2) Get undistorted image
	undist_rgb_img = undistort_image(rgb_img, objpoints, imgpoints)
	# 3) Convert rgb image into hls space
	hls_img = cv2.cvtColor(undist_rgb_img, cv2.COLOR_RGB2HLS)
	# 4) Convert the original img from RGB to Gray scale
	gray_img = cv2.cvtColor(undist_rgb_img, cv2.COLOR_RGB2GRAY)
	# 5) Threshold x gradient
	grad_x_binary = abs_sobel_thresh(gray_img, orient = 'x', sobel_kernel = 3, thresh=(20, 100))
	# 6) Threshold s channel in hls space
	color_s_binary = color_threshold(hls_img, channel = 2, thresh = (170, 255))
	# 7) Combine the two binary thresholds
	combined_binary = np.zeros_like(grad_x_binary)
	combined_binary[(color_s_binary == 1) | (grad_x_binary == 1)] = 1
	# 8) Get unwarp_image from undist_img using perspective_matrix
	unwarp_img = unwarp_image(combined_binary, perspective_matrix)
	return unwarp_img


def slide_window_polyfit(binary_warped):
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	# out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]//nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window + 1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	return left_fit, right_fit


def pre_polyfit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & 
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & 
                       (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit_new, right_fit_new = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new


def draw_lanes(img, binary_warped, left_fit, right_fit, Minv):

    new_img = np.copy(img)
    new_img = cv2.resize(new_img, (720, 405))
    if len(left_fit) == 0 or len(right_fit) == 0:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h, w = binary_warped.shape
    ploty = np.linspace(0, h - 1, num = h)# to cover same y-range as image
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 100, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed = False, color=(255, 255, 0), thickness = 20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed = False, color=(255, 255, 0), thickness = 20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 

    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def cal_curvature(binary_warped, left_fit, right_fit):
	y_eval = np.max(binary_warped.shape[1] - 1)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	return left_curverad, right_curverad


def cal_off_center(binary_warped, left_fit, right_fit):
	# Assume lane width is 3.7 meters and it counts for 475 pixels in the image
	# Calculate meters per pixel	
	meter_per_pix = 3.7/475 
	car_position = binary_warped.shape[1]/2
	h = binary_warped.shape[0]
	left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
	right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
	lane_center_position = (left_fit_x_int + right_fit_x_int) / 2
	center_dist = (car_position - lane_center_position) * meter_per_pix
	return center_dist


def draw_data(img, curvature, center_dist):
	new_img = np.copy(img)
	h = new_img.shape[0]
	font = cv2.FONT_HERSHEY_PLAIN
	cv2.putText(new_img, 'Automatic Lane Detection', (20, 60), font, 2, (0, 255, 255), 2, cv2.LINE_AA)

	text = 'Curve radius: ' + '{:04.2f}'.format(curvature) + 'm'
	cv2.putText(new_img, text, (20, 100), font, 2, (0, 255, 255), 2, cv2.LINE_AA)
	direction = ''
	if center_dist > 0:
		direction = 'right'
	elif center_dist < 0:
		direction = 'left'
	abs_center_dist = abs(center_dist)
	text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' off center'
	cv2.putText(new_img, text, (20,140), font, 2, (0, 255, 255), 2, cv2.LINE_AA)
	return new_img


if __name__ == '__main__':

	cal_images_dir = 'camera_cal/'
	# Calculate iamge points and object points needed for camera calibration
	imgpoints, objpoints = get_calibration_points(image_dir = cal_images_dir)

	# Define source points and destination points for perspective transform
	apex, apey = 360, 258
	offset_far = 50
	offset_near = 10
	src_points = np.float32([[int(apex - offset_far), apey],
		[int(apex + offset_far), apey],
		[int(0 + offset_near), 390],
		[int(720-offset_near), 390]])
	dst_points = np.float32([[0, 0], [720, 0], [0, 405], [720, 405]])
	# Get perspective_matrix and perspective_matrix_inv from src_points and dst_points
	perspective_matrix, perspective_matrix_inv = get_unwarp_matrix(src_points, dst_points)
	# Load video from mp4
	cap = cv2.VideoCapture('challenge_video.mp4')
	# Initialize left lane fit line and right lane fit line
	left_fit = []
	right_fit = []
	# Loop through the video frames
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# Apply image processing pipeline 
			binary_warped = process_image_pipeline(rgb_img, objpoints = objpoints, imgpoints = imgpoints, perspective_matrix = perspective_matrix)
			out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
			# If no pre-exist fit, apply sliding windows to fit polyline
			if  len(left_fit) == 0 or len(right_fit) == 0:
				left_fit, right_fit = slide_window_polyfit(binary_warped = binary_warped)
			# if pre-exist fit, skip sliding window
			else:
				left_fit, right_fit = pre_polyfit(binary_warped = binary_warped, left_fit_prev = left_fit, right_fit_prev = right_fit)

			img_out = draw_lanes(rgb_img, binary_warped, left_fit = left_fit, right_fit = right_fit, Minv = perspective_matrix_inv)
			left_curverad, right_curverad = cal_curvature(binary_warped, left_fit, right_fit)
			curvature = (left_curverad + right_curverad) / 2
			center_dist = cal_off_center(binary_warped, left_fit, right_fit)
			img_out = draw_data(img_out, curvature, center_dist)

			cv2.imshow('image', img_out)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

