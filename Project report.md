Airbus Ship Detection
====
Project proposal
---
This is a challenge on Kaggle which is to build a model that detects all ships in satellite images as quickly as possible. 

Behind this challenge, there is a backstory: Shipping traffic is growing fast. More ships increase the chances of infractions at sea like environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement. This has compelled many organizations, from environmental protection agencies to insurance companies and national government authorities, to have a closer watch over the open seas.

Airbus offers comprehensive maritime monitoring services by building a meaningful solution for wide coverage, fine details, intensive monitoring, premium reactivity and interpretation response. Combining its proprietary-data with highly-trained analysts, they help to support the maritime industry to increase knowledge, anticipate threats, trigger alerts, and improve efficiency at sea.

A lot of work has been done over the last 10 years to automatically extract objects from satellite images with significative results but no effective operational effects. Now Airbus is turning to Kagglers to increase the accuracy and speed of automatic ship detection.

We now divide this challenge into three steps: First, we use two models to detect whether there is a ship in the image. Secondly, we need to mark the location of ships. At last, we may design an interface for users to upload images and dectect ships.

Theory of model
---
### VGG16
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.

![image](https://github.com/MuweiZ/machine-learning-project/VGG16 figure1.png)

* The Architecture

The input to cov1 layer is of fixed size 224 x 224 RGB image. The image is passed through a stack of convolutional (conv.) layers, where the filters were used with a very small receptive field: 3×3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations, it also utilizes 1×1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1-pixel for 3×3 conv. layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv.  layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2×2 pixel window, with stride 2.

![image](https://github.com/MuweiZ/machine-learning-project/VGG16 Architecture figure1.png)

Three Fully-Connected (FC) layers follow a stack of convolutional layers (which has a different depth in different architectures): the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer. The configuration of the fully connected layers is the same in all networks.

All hidden layers are equipped with the rectification (ReLU) non-linearity. It is also noted that none of the networks (except for one) contain Local Response Normalisation (LRN), such normalization does not improve the performance on the ILSVRC dataset, but leads to increased memory consumption and computation time.

* Configurations

The ConvNet configurations are outlined in figure below. The nets are referred to their names (A-E). All configurations follow the generic design present in architecture and differ only in the depth: from 11 weight layers in the network A (8 conv. and 3 FC layers) to 19 weight layers in the network E (16 conv. and 3 FC layers). The width of conv. layers (the number of channels) is rather small, starting from 64 in the first layer and then increasing by a factor of 2 after each max-pooling layer, until it reaches 512.

![image](https://github.com/MuweiZ/machine-learning-project/VGG16 Configurations figure1.png)

### U-Net

U-Net is considered one of the standard CNN architectures for image classification tasks, when we need not only to define the whole image by its class but also to segment areas of an image by class, i.e. produce a mask that will separate an image into several classes. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.

The network is trained in end-to-end fashion from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC), U-Net won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512×512 image takes less than a second on a modern GPU.

* The Architecture

![image](https://github.com/MuweiZ/machine-learning-project/U-Net Architecture figure1.png)

The network architecture is illustrated in figure above. It consists of a contracting path (left side) and an expansive path (right side). The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3×3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2×2 max pooling operation with stride 2 for downsampling.

At each downsampling step, feature channels are doubled. Every step in the expansive path consists of an upsampling of the feature map followed by a 2×2 convolution (up-convolution) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3×3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution.

* Configurations

The input images and their corresponding segmentation maps are used to train the network with the stochastic gradient descent. Due to the unpadded convolutions, the output image is smaller than the input by a constant border width. A pixel-wise soft-max computes the energy function over the final feature map combined with the cross-entropy loss function. The cross-entropy that penalizes at each position is defined as:

![image](https://github.com/MuweiZ/machine-learning-project/U-Net formula figure1.png)

The separation border is computed using morphological operations. The weight map is then computed as:

![image](https://github.com/MuweiZ/machine-learning-project/U-Net formula figure2.png)

where wc is the weight map to balance the class frequencies, d1 denotes the distance to the border of the nearest cell and d2 denotes the distance to the border of the second nearest cell.

Training
---
### VGG16

*	Data processing

*	Split dataset

*	Combination mask with original pictures

*	Build Model

* Train the data and get the loss history

### U-Net

*	Data processing

At first, we process the data and get the model parameters.

dp figure

*	Split dataset

We set all the data to train and validation set.

20% of all the images are selected to be validation dataset.

*	Combination mask with original pictures

decold all rlus in to images

find and mask the ships

![image](https://github.com/MuweiZ/machine-learning-project/combination figure1.png)

*	Build Model

Build a U-Net model to fit the dataset

![image](https://github.com/MuweiZ/machine-learning-project/model pic1.png)

![image](https://github.com/MuweiZ/machine-learning-project/model pic2.png)

* Train the data and get the loss history

Train the model, let the training processing break down when loss is low enough.

Finally, we get loss function

![image](https://github.com/MuweiZ/machine-learning-project/train pic1.png)

![image](https://github.com/MuweiZ/machine-learning-project/train pic2.png)

* Predict and visualize

Use the U-Net model to make some predictions on test dataset

Then mask the predicted ships on the map

Finally compare the predicted ships with the ground truth

![image](https://github.com/MuweiZ/machine-learning-project/Predict pic1.png)

Result analysis and comparison
---
### VGG16

### U-Net

We can see from the output that when the background is only sea, the ship can be clearly detected.

However, if there are not only boats but also sea land, wharf, or clouds in the background, then the recognition rate of this model is greatly reduced. It would recognize clouds and land as ships too.

### Comparison









