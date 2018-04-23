# Self-Driving Car Engineer Nanodegree Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients.
* Apply a distortion correction to raw images.
* Use gradients and color thresholding etc., to create a binary image that better represents the lanes.
* Apply a perspective transform to unwarp the binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[im01]: ./examples/gbao_calibration.png "Chessboard Calibration"
[im02]: ./examples/gbao_calibration_result.png "Calibration Result"



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


The image below illustrate the result of applying `undistort_image`, using the calibration and distortion coefficients to one of the chessboard images:
![alt text][im02]


