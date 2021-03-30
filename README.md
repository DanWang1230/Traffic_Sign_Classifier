# **Traffic Sign Recognition** 

The goals/steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

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

#### 1. Basic summary of the data set

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

This is a bar chart showing how the data is distributed: blue for the training data set and yellow for the validation.

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Image data preprocessing

As a first step, I decided to convert the images to grayscale because color is not an important feature in the project.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1] ![alt text][image2]

As a last step, I normalized the image data so the data has mean zero and equal variance and the position of the image doesn't matter.

To increase the accuracy of the validation set, I decided to change the arichitecture of the LeNet network from the lecture.


#### 2. Model architecture

A LeNet-5 architecture is chosen for this task. Using the original LeNet-5, I achieved a high accuracy on the training set but low accuracy on the validation set (around 0.89). To solve for the overfitting problem, I added dropout layers after the fully connected layers. 

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


#### 3. Model training

To train the model, I used the Adam optimizer, the batch size of 128, 50 epochs, the learning rate of 0.001, and the keep probability of 0.5 for dropout.

#### 4. Performance

* training set accuracy of 0.998
* validation set accuracy of 0.967 
* test set accuracy of 0.947

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web

Here are five German traffic signs that I found on the web after resizing and grayscaling:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

They all are under well lighting condition and pretty clear and should not be difficult to classify.

#### 2. Model's predictions on new traffic signs
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic signal      	| Traffic signal 								| 
| Wild animal crossing  | Wild animal crossing 							|
| No entry				| No entry          							|
| Pedestrians	   		| Pedestrians					 				|
| Stop          		| Stop               							|


The model was able to correctly guess all the five traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.7%.

#### 3. Determine how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 19th cell of the notebook.

For the first image, the model is 100% sure that this is a traffic signal(probability of 1), and the image indeed is a traffic signal. And the other four softmax probabilities are close to 0. This is also the case for the other four new images I obtained from web.


