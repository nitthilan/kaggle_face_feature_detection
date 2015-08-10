Facial Keypoint Detection [Kaggle]
==================================
This is a approach to understand the Facial Keypoint Detection problem explained in Kaggle using torch7/lua

Reference:
	- https://www.kaggle.com/c/facial-keypoints-detection
	- http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

Details of data:
================
Input:
	- Train:
		- 30 features for facial key points or 15 pairs of (x,y) position
		- 7049 images
	- Test:
		- 1783 images
		- Each image there are a set of features which has to be calculated and submitted

Understanding Input:
- Num images with all the 30 feature vectors 	2140

- Num images for each feature:
	left_eye_center_x	7039	
	left_eye_center_y	7039	
	right_eye_center_x	7036	
	right_eye_center_y	7036	
	left_eye_inner_corner_x	2271	
	left_eye_inner_corner_y	2271	
	left_eye_outer_corner_x	2267	
	left_eye_outer_corner_y	2267	
	right_eye_inner_corner_x	2268	
	right_eye_inner_corner_y	2268	
	right_eye_outer_corner_x	2268	
	right_eye_outer_corner_y	2268	
	left_eyebrow_inner_end_x	2270	
	left_eyebrow_inner_end_y	2270	
	left_eyebrow_outer_end_x	2225	
	left_eyebrow_outer_end_y	2225	
	right_eyebrow_inner_end_x	2270	
	right_eyebrow_inner_end_y	2270	
	right_eyebrow_outer_end_x	2236	
	right_eyebrow_outer_end_y	2236	
	nose_tip_x	7049	
	nose_tip_y	7049	
	mouth_left_corner_x	2269	
	mouth_left_corner_y	2269	
	mouth_right_corner_x	2270	
	mouth_right_corner_y	2270	
	mouth_center_top_lip_x	2275	
	mouth_center_top_lip_y	2275	
	mouth_center_bottom_lip_x	7016	
	mouth_center_bottom_lip_y	7016	

Model:
	- Assuming a linear 1D model with 96x96(9216) inputs
	- Without having non-linear layers inbetween we are actually creating a single layer effectively. So a ReLU is introduced inbetween Linear layers
	- Since this problem is a Linear Regression problem i.e. the output data can take any values between (0 to 96 or 0 to 1 in normalised form) we use MSE (Mean Square Error) Criterion

Normalising the input:
	- The input data had to be scaled by 255 to map it between (0,1)
	- Feature points scaled by 96 (image width) to map it between 0,1.
	- Any Feature which is not present is currently initialised with -1
	- the idea is to map them back to what the network generates thus making the error zero during the criterion:forward pass
	- this had to be done since the forward passes due to 96x96x100 + 100*30 weights scaled addition results in NaN

initialising the weights:
	- The default way of initialisng the weights is given by 
		- stdv is the 1.0/sqrt(Num Weights in the layer)
		- self.weight:uniform(-stdv, stdv)
      	- self.bias:uniform(-stdv, stdv)
    - One good reference for initialisation: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
	- Could try other modules like https://github.com/e-lab/torch-toolbox/tree/master/Weight-init ??


Training:
	- Splitting the train data of 2140 into 80-20 for learning and testing
	- Current algorithm used for training gradient descent is Nesterov’s Accelerated Gradient Descent (NAG) and the optimisation parameters used are
	- sgd_params = { learningRate = .01, learningRateDecay = .001, weightDecay = 0, momentum = .9 }

Results:
	- With the 80-20 split the current validation to train loss ratio is around 63.1607208073 i.e. training loss is  0.00087948051525779 and the validation/test set loss is 0.055548623279652.
	- The issue was a coding error. Train loss was in MSE while current loss was norm
	- 247 current loss = 0.026864869890844	
	247current validation loss 0.056043914015213

Points to remember:
	- convolution networks expects input to be a 3d or 4d vector and not just a 2d vector so add 1x96x96

Questions to be answered:
	- storing and restoring models
	- using batch mode for updation instead of sgd
	- using random image index to access
	- using all the input images instead of only images which have all the 30 features
	- using data agumentation i.e. increasing the number of data inputs by doing image horizontal flips and vertical flips