# **Traffic Sign Recognition** 
## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Results/original_image.jpg
[image2]: ./Results/gray.jpg
[image3]: ./Results/class_distribution.png
[image4]: ./Results/template_0.jpg
[image5]: ./Results/template_1.jpg
[image6]: ./Results/template_2.jpg
[image7]: ./Results/template_3.jpg
[image8]: ./Results/template_4.jpg


---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculated the summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed: blue for the training data set and yellow for the validation.

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color is not an important feature in the project.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1] ![alt text][image2]

As a last step, I normalized the image data so the data has mean zero and equal variance and the position of the image doesn't matter.

To increase the accuracy of the validation set, I decided to change the arichitecture of the LeNet network from the lecture.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16	    |
| Flatten				| outputs 400									|
| Fully connected		| outputs 120 									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 84 									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 43									|
| Dropout				|												|
| Softmax				|      									        |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer, the batch size of 128, 50 epochs, the learning rate of 0.001, and the keep probability of 0.5 for dropout.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.967 
* test set accuracy of 0.947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The same architecture as in the lecture. It's a good starting point.
* What were some problems with the initial architecture?
The validation set accuracy is low (around 0.89).
* How was the architecture adjusted and why was it adjusted?
I chose a different architecture and added dropout layers after the fully connected layers to solve for the overfitting problem: a high accuracy on the training set but low accuracy on the validation set.
* Which parameters were tuned? How were they adjusted and why?
The ephoch number is increased to reach to an optimum. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Adding the dropout layers helps increase the model accuracy by handling overfitting.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet-5
* Why did you believe it would be relevant to the traffic sign application?
We learned it from the lecture, and it works well with the digit classfication application.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The training set accuracy 0.998 is high, which is expected. The validation set accuracy 0.967 is fair enough. The trained CNN gives an accuracy of 0.947 for the unseen test set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web after resizing and grayscaling:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

They all are under well lighting condition and pretty clear and should not be difficult to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic signal      	| Traffic signal 								| 
| Wild animal crossing  | Wild animal crossing 							|
| No entry				| No entry          							|
| Pedestrians	   		| Pedestrians					 				|
| Stop          		| Stop               							|


The model was able to correctly guess all the five traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the notebook.

For the first image, the model is 100% sure that this is a traffic signal(probability of 1), and the image indeed is a traffic signal. And the other four softmax probabilities are close to 0. This is also the case for the other four new images I obtained from web.


