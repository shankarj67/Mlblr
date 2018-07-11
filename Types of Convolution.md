# Convolution 

In terms of Computer vision Convolution is nothing but a dot product of input image and Filter which is used to extract the features from the input image, let us clarify this using gif

![alt text](https://cdn-ak.f.st-hatena.com/images/fotolife/k/kaeken/20161113/20161113135225.gif)

Behind the scene

![alt text](https://cdn-images-1.medium.com/max/1600/1*ZCjPUFrB6eHPRi4eyP6aaA.gif)

What's going on behind the scene is we are basically doing element-wise multiplication between a patch of input image and filter

Now that we know what is a convolution is, let's look into the different types of convolution:

# Dilated Convolution (a.k.a. atrous convolutions):

In the paper [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/pdf/1511.07122.pdf), It introduces another parameter to convolutional layers called the dilation rate which is based on the fact that dilated convolutions support an exponential expansion of the receptive field without loss of resolution or coverage.

![alt text](https://mlblr.com/images/dilated.gif)

In simpler words, the main idea is to defines a spacing or say fill the added pixel with zeroes and then compute a convolution, Suppose A 3x3 kernel with a dilation rate of 2 will have the same field of view as a 5x5 kernel, see this image for proper understanding with the different dilated rate

![alt text](https://i.stack.imgur.com/tOg0g.png)

The above shows dilated convolution where Red dots are the inputs to a filter which is 3x3 here, and blue area is the receptive field captured by each of these inputs. 


The main goal of dilated convolution is to:
1. Increase the size of the receptive field without any loss of dimension of an image
3. Increasing the size of receptive field gives more information and better feature map.
4. Computation and memory cost is low with larger receptive field
2. This convolution has better performance as described here in this paper [paper link](https://arxiv.org/pdf/1511.07122.pdf)


Quick recap:
1. Dilated convolution delivers a wider view at the same computational cost.
2. Dilated convolutions are popular mostly in real time segmentation which cares about wide field view and can't afford multiple convolutions or larger kernels.



# 1X1 CONVOLUTION or pointwise:

![alt text](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif)

1x1 convolutions were first proposed at [Network-in-network(NiN)](https://arxiv.org/pdf/1312.4400v3.pdf) which says that this convolution does not consider spatial information but takes channels into account.

To put it simply, we can say this is the only convolution which is used to reduce the depth of input image which is also called channel in 3D volume, thus speeding up the computation. let us clarify by looking into below image 

![alt text](https://qph.ec.quoracdn.net/main-qimg-086e3e0481a94f53e7ebd6bc39f282c9.webp)

In the above image we are using a [1x1x32] filter and convolving on [28x28x192], the [1x1x32] conv layer transfer the input layer with [192] channel into an output of [28x28x32], which reduces the channel of the input image and it also helps to speed up the computation.

Features of 1x1 Convolution:
1. It helps to reduce the dimensionality
2. It adds non-linearity to the feature map which enhances the representation of a network
3. It is highly used in inception nets architecture where it reduces the dimension of the previous layer for faster computation


# Maxpooling:

Max polling reduces the dimensionality of the image by reducing the number of pixels in the feature map.

![alt text](https://mlblr.com/images/maxpool.gif)

The main idea behind max pooling is that we are selecting only the maximum value from the pixel which is more activated and discarding all the other values which are not activated.


Max-pooling will always apply after the convolution is done means we always apply max pooling on the feature map which is obtained after convolution in order to reduce the dimension, let us clarify by looking into below gif

![alt text](https://saitoxu.io/images/2017-01-01-convolution-and-pooling.png)

In the above picture, we have an Input image of 11x11 pixels.

Convolution layer is applied with a filter of [3x3] which gives [9x9] feature map or output image.

Now, at the last max pooling layer is applied on [9x9] image which reduces the dimension of the image to [3x3] which has a receptive field of the whole image.

Why max-pooling?

1. It helps us to reduce the parameter within the model  - this is also called down-sampling
2. It reduces the size of the image because of the less parameter and It can still identify the whole image.
3. It also helps in reducing the computation time








