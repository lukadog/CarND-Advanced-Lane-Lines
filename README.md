# Self-Driving Car Engineer Nanodegree Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients.
* Undistort the raw images.
* Use combination of gradients and color thresholding etc., to create a binary image that better represents the lanes.
* Apply perspective transform to warp the binary image to top down view (aka "birds-eye view").
* Detect pixels that belong to the lanes using a sliding window technique and fit them into a polyline.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane back onto the original image.
* Display visual display of the lane lines, pavement and lane curvature and vehicle position.

[//]: # (Image References)

[im01]: ./examples/gbao_calibration.png "Chessboard calibration"
[im02]: ./examples/gbao_calibration_result.png "Calibration result"
[im03]: ./examples/gbao_unwarp.png "Image processing pipeline"
[im04]: ./examples/gbao_unwarp_lane_binary.png "Result of warp iamge"
[im05]: ./examples/gbao_combine.png "Generate binary image"
[im06]: ./examples/visualization_slide_window.png "Poly fit"
[im07]: ./examples/gbao_result.png "Draw result"
[im08]: ./examples/find_lane.gif "gif"

### Camera calibration:

Camera calibration is handled by `get_calibration_points` function and `undistort_image` function, which are comprised of following steps:

* Load calibration chessboard images
* Convert RGB/BGR chessboard images to gray scale
```
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
```
* Finding chessboard corners (for an 9x6 board)

```
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
```
* As a result, a set of image points and objects points are returned from `get_calibration_points`

* `undistort_image` uses mage points and objects points to calculate the camera matrix.
```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

```

![alt text][im01]

*Note: some of the calibration images don't cover all inner corners, so when using these images for calibration, the cv2.findChessboardCorners returns false. Therefore, I have to identify those bad images and delete them.*

```
calibration_dir = './camera_cal/'
fname = 'calibration'
for i in range(1, 20):
	image_dir = calibration_dir + fname + str(i) + '.jpg'
	img = cv2.imread(image_dir)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # Identify the bad calibration images
	if ret == False:
		print(i)
```


The image below illustrates the result of applying `undistort_image`, using the calibration and distortion coefficients to one of the chessboard images:
![alt text][im02]

### Image processing pipeline:

The image processing pipeline is handled by `process_image_pipeline`, which is comprised of below steps:
* Convert BGR image into RGB (depends how you read the image)
* Resize the image to consistent size so it's invariant to raw input images.
* Undistort the image using calibration matrix
* Generate binary image using x direction gradient and S chanel threshold of HLS color space. A result of binary image is shown below:
![alt text][im05]

```
	grad_x_binary = abs_sobel_thresh(gray_img, orient = 'x', sobel_kernel = 3, thresh=(20, 100))
	color_s_binary = color_threshold(hls_img, channel = 2, thresh = (170, 255))
	combined_binary = np.zeros_like(grad_x_binary)
	combined_binary[(color_s_binary == 1) | (grad_x_binary == 1)] = 1
```

* Perspective transformation and warp the binary image to top down view

```
	unwarp_img = unwarp_image(combined_binary, perspective_matrix)
```

![alt text][im03]

Below shows a result of the output of the image processing pipeline

![alt text][im04]


### Identify the lane line:

The lane lines are identified by `slide_window_polyfit` and `pre_polyfit` functions. If there is no pre-identified lanelines, `slide_window_polyfit` is triggered by applying sliding window, otherwise, `pre_polyfit` is used, which requires less computation.

![alt text][im06] shows the visualization of sliding window and polyline fitting.


### Calculate curvature radius and car position:

* carvature radius is calculated by `cal_curvature` function.
```
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
```
The final answer would be average of the two lanes.

* car position is calculated by `cal_off_center` function:
Note: I am assuming the lane width is 3.7 meters and I measured the pixels between lane lines, which are 475 pixels.
```
	meter_per_pix = 3.7/475 
	car_position = binary_warped.shape[1]/2
	h = binary_warped.shape[0]
	left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
	right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
	lane_center_position = (left_fit_x_int + right_fit_x_int) / 2
	center_dist = (car_position - lane_center_position) * meter_per_pix
```

### Draw result:
The result is drawn by the function `draw_data`. Result is shown below:

![alt text][im07]

### Discussion:

1. The input video sometimes of different dimensions, so I resize them to keep size consistent.
2. When plotting binary images, we need to convert it to int8, otherwise it won't show.
3. Sliding window is really slow, but after I skip it for subsequent frames, performance is slightly better.
4. The curvature radius estimation is not very stable, I guess part of the reason is that the road is not flat so the perspective matrix doesn't work well for every image.

### Final result:

![alt text][im08]


