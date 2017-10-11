# **Behavioral Cloning** 

## Writeup

*Behavioral Cloning Project*

The goals / steps of this project are the following:
* Use the simulator to collect data of a good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track number 1 without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture.png "Model Visualization"
[image2]: ./images/image1.jpeg "Data sample 1st track"
[image3]: ./images/image2.jpeg "Data sample 2nd track"
[image4]: ./images/image3.jpeg "Data sample. Flipped images"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* WRITEUP.md summarizing the results
* video.mp4 a video recording of the vehicle driving autonomously
* 2nd_track_video.mp4 a drive on the 2nd track

#### 2. Submission includes functional code

Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The project uses a CNN model proposed by NVIDIA in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

It consists of a convolution neural network with 5x5 and 3x3 filters and depths between 24 and 64 (model.py lines 58-62) 

Convolution layers use RELU as activation function to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 57). 

#### 2. Attempts to reduce overfitting in the model

The training harness allows loading of multiple simulator logs model so that it can be trained and validated on different data sets. That helps to ensure that the model doesn't overfit (code line 98). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate is not tuned manually (model.py line 69).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
To gather more generalized data, I was recording driving on different tracks, as well as driving in different directions. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to collect enough data (different tracks, driving in both directions) and test the trained model.

My first step was to use a convolution neural network model similar to LeNet, I thought this model might be appropriate because it was fast to train and test results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I decided to augment train data set. For augmenting data, I took images from all 3 cameras and added a flipped version for each image.

Then, after re-training LeNet with augmented data set, the car drove better but was not able to drive through sharp corners. I concluded that LeNet is too simple to learn all important features of the track and I decided to try the more sophisticated model, as one proposed by NVIDIA for their autonomous driving projects.  

The final step was to train chosen architecture with augmented data set and run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

As an extra exercise, I tried to train the model on the full set (without splitting data into train and validation sets) and gave it a try on the second track. The car was able to go through that track as well.

#### 2. Final Model Architecture

Here is a visualization of the architecture (model.py lines 58-67)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by my experiments. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Extras

#### Driving on the 2nd track

The trained model was able to drive the vehicle on the 2nd track as well. A video demonstrating that can be found in 2nd_track_video.mp4 or at https://youtu.be/oE6o5JIf6zs
