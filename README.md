# SelfDrivingCar

My First real project using deep learning using Tensorflow and keras with a convolutional neural networks

![](vid/video.gif)

## Model Architecture and Training Strategy

I train the model using a convolutional neural network predicting one linear output. first of all to reduce the overfitting, I used data augmentation by flipping each image of the dataset and by using the left/right camera. The model used an Adam optimizer For the training data, I used  the central, left and right camera, each one, randomly flip.

## Final Model Architecture

The architecture is the following:
<ul>
    <li>One convolution of 8 filters (9*9) [Elu activation]</li>
    <li>One convolution of 16 filters (5*5) [Elu activation]</li>
    <li>One convolution of 32 filters (4*4) [Elu activation]</li>
    <li>Max pooling pool_size=[2,2]</li>
    <li>Flatten layer</li>
    <li>Dropout: 0.5</li>
    <li>Fully conected: 1024 [Elu]</li>
    <li>Dropout: 0.3</li>
    <li>Fully conected: 1 (Linear output)</li>
</ul>
