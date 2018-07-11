
# Convolution

After Completing this article, You will be able to answer

1. What is convolution?
2. The math behind the Convolution.
3. What makes the CNN so popular?
4. What is Filter?
5. how to write an edge detector using python.
3. How to write a basic CNN in keras using python


Before deep dive into Convolution, Let us first build the idea of how the computer sees the object by looking into this image

![alt text](https://static.commonlounge.com/fp/600w/pZdmRKkodLhyFKbusUMSQwKpl1520490136_kc)

Just think for a moment that an image is just a number between 0-255, then how the computer algorithm detects an edge, pattern or even an object like birds, car etc.
If you are confused about all of this then you are in the right place. By the end of this article, we will have the intuition of how computer detects an object.

Let's define what is convolution.

![alt text](https://cdn-ak.f.st-hatena.com/images/fotolife/k/kaeken/20161113/20161113135225.gif)

First, let me clarify one thing that here we are talking about Convolution neural network which is different from any other neural network.

In terms of Computer vision Convolution is nothing but a dot product of input image and Filter, let us clarify this using gif

![alt text](https://cdn-images-1.medium.com/max/1600/1*ZCjPUFrB6eHPRi4eyP6aaA.gif)


In simple words, what happens is: the filter which is of yellow colour is moving in the input from left to right and top to bottom and each value of filter is multiplied by the value of input which is of green colour on the same position. The result which is obtained by multiplication are then summed up and output is generated which is of Pink colour on the right side of an image, the output is also called Feature map.

What's going on behind the scene is we are basically doing element-wise multiplication between a patch of input image an filter

If you are still confused about filter then let me tell you filters are used to detect some pattern such as edge, horizontal line etc. It depends on you what you want to detect from an  image and on the basis of that you will define your filter

![alt text](https://media1.tenor.com/images/0188c63209aced59f1583e1ca94e509e/tenor.gif?itemid=3550689)

Even if you are not convinced what filter is don't worry we will go deeper into the filter in our next article



# Filter

Till now we have an idea about what is convolution and little brief about a filter, let's dive deep into a filter and see how we can make the best use of a filter in CNN.

A filter in a CNN is like a matrix with which we multiply an image of same input size to get the feature map or we can also say in simpler words: Filters are what detects the pattern. 

No of filters =No of the feature map


We can produce different feature map by applying a different filter such as edge detector and many more.

There are many filters available which are really useful in computer vision task but one of the most important one is edge detection. Edge detection aims to identify the pixels of an image which has higher brightness. let's apply one of the simplest edge detector to our image and see the result

Here is the kernel
Kernels
   |         
| ------------- |:-------------:| -----:|
| -1     | -1 | -1 |
| -1      | 8      |   -1 |
| -1 | -1      |    -1|

Here is our original image:

![alt text](http://machinelearninguru.com/_images/topics/computer_vision/basics/convolution/sharpen.jpg)


Let's write the python code to detect edge by applying the above kernel

```python

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io, color
from skimage import exposure
img = io.imread('image.png')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# we use 'valid' which means we do not add zero padding to our image
edges = scipy.signal.convolve2d(img, kernel, 'valid')
print( '\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255)
# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)
plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
```
After applying the kernel on image, this is how our image look like

![alt text](http://machinelearninguru.com/_images/topics/computer_vision/basics/convolution/edges.jpg)




# Convolution neural network
Now that we build up the foundation of convolution and filter, let us quickly wrap up by defining a little bit about Convolution neural network

![Lenet Architecture](https://cdn-images-1.medium.com/max/1600/1*8Ut7fQHswfO2zZngh6BYfg.png)

Convolution neural network also known as CNN has the ability to detect the patterns and edges which makes the CNN most popular for computer vision task. It's because of Convolution only CNN can detect edges, patterns and even an object like a car, bird etc. this idea of convolution comes after a lot of research and hard work which we are using right now.


If we see the above image of CNN, you can see that there are lots of layers but I want you to put focus on conv layer which contains a feature map which we obtained from the operation of convolution and filter and then this feature map is passed over to next layer to detect more rich feature, if I have to sum everything up let me show you a photo which clears everything that we have covered

![Feature extractor](https://cdn-images-1.medium.com/max/880/1*Ji5QhY9QXBlpNNLH4qAcNA.png)


We can clearly see that CNN can easily detect from edge to car with the help of underlying principle, Although there are a lot of things going on behind the scene like max pooling, Activation function etc but Convolution and filters are the core of Convolution neural network.




The code below is for CNN Structure what we discussed above:

```python

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

```
Let's quickly look into the convolution code :



```python

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
                 
```
The first argument passed is Conv2D() layer function which is output channel - in this case, we have 32 channels means we will get 32 feature map, The next input is the Filter size which is chosen to be a 5x5 moving window, don't worry about the rest for now.

Now that you have an idea that with one image we can have 32 feature map means 32 different output, this is the main concept behind CNN which helps to detect an object like car, bird etc.


A quick recap of what we covered:

1. What are convolution and filter?
2. Why CNN is so popular in computer vision task.
3. How to write an edge detector using python
4. See the basics of Convolution code in keras using python.

Hope you like this article :)






