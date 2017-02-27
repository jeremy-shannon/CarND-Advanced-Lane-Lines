# Self-Driving Car Engineer Nanodegree Program
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[im01]: ./output_images/01-calibration.png "Chessboard Calibration"
[im02]: ./output_images/02-undistort_chessboard.png "Undistorted Chessboard"
[im03]: ./output_images/03-undistort.png "Undistorted Dashcam Image"
[im04]: ./output_images/04-unwarp.png "Perspective Tansform"
[im05]: ./output_images/05-colorspace_exploration.png "Colorspace Exploration"
[im06]: ./output_images/09-sobel_magnitude_and_direction.png "Sobel Magnitude & Direction"
[im07]: ./output_images/11-hls_l_channel.png "HLS L-Channel"
[im08]: ./output_images/12-lab_b_channel.png "LAB B-Channel"
[im09]: ./output_images/13-pipeline_all_test_images.png "Processing Pipeline for All Test Images"
[im10]: ./output_images/14-sliding_window_polyfit.png "Sliding Window Polyfit"
[im11]: ./output_images/15-sliding_window_histogram.png "Sliding Window Histogram"
[im12]: ./output_images/16-polyfit_from_previous_fit.png "Polyfit Using Previous Fit"
[im13]: ./output_images/17-draw_lane.png "Lane Drawn onto Original Image"
[im14]: ./output_images/18-draw_data.png "Data Drawn onto Original Image"

[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first two code cells of the Jupyter notebook `project.ipynb`.  

The OpenCV functions `findChessboardCorners` and `calibrateCamera` are the backbone of the image calibration. A number of images of a chessboard, taken from different angles with the same camera, comprise the input. Arrays of object points, corresponding to the location (essentialy indices) of internal corners of a chessboard, and image points, the pixel locations of the internal chessboard corners determined by `findChessboardCorners`, are fed to `calibrateCamera` which returns camera calibration and distortion coefficients. These can then be used by the OpenCV `undistort` function to undo the effects of distortion on any image produced by the same camera. Generally, these coefficients will not change for a given camera (and lens). The below image depicts the corners drawn onto twenty chessboard images using the OpenCV function `drawChessboardCorners`:

![alt text][im01]

*Note: Some of the chessboard images don't appear because `findChessboardCorners` was unable to detect the desired number of internal corners.*

The image below depicts the results of applying `undistort`, using the calibration and distortion coefficients, to one of the chessboard images:

![alt text][im02]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The image below depicts the results of applying `undistort` to one of the project dashcam images:

![alt text][im03]

The effect of `undistort` is subtle, but can be perceived from the difference in shape of the hood of the car at the bottom corners of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I explored several combinations of sobel gradient thresholds and color channel thresholds in multiple color spaces. These are labeled clearly in the Jupyter notebook. Below is an example of the combination of sobel magnitude and direction thresholds:

![alt text][im06]

The below image shows the various channels of three different color spaces for the same image:

![alt text][im05]

Ultimately, I chose to use just the L channel of the HLS color space to isolate white lines and the B channel of the LAB colorspace to isolate yellow lines. I did not use any gradient thresholds in my pipeline. I did, however finely tune the threshold for each channel to be minimally tolerant to changes in lighting. As part of this, I chose to normalize the maximum values of the HLS L channel and the LAB B channel (presumably occupied by lane lines) to 255, since the values the lane lines span in these channels can vary depending on lighting conditions. Below are examples of thresholds in the HLS L channel and the LAB B channel:

![alt text][im07]
![alt text][im08]

*Note: the B channel was not normalized if the maximum value for an image was less than 175, signifying that there was no yellow actually in the image. Otherwise the resulting thresholded binary image was very noisy.*

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is clearly labeled in the Jupyter notebook, in the seventh and eighth code cells from the top.  The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

```
I had considered programmatically determining source and destination points, but I felt that I would get better results carefully selecting points using one of the `straight_lines` test images for reference and assuming that the camera position will remain constant and that the road in the videos will remain relatively flat. The image below demonstrates the results of the perspective transform: 

![alt text][im04]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

