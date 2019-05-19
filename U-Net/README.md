U-Net
===========

U-Net technology
-------
  U-Net is one of the earliest semantics segmentation algorithms using full convolution network. The symmetrical U-shaped structure including compressed path and extended path was very innovative at that time, and to some extent affected the design of the following segmentation networks. The name of the network is also taken from its U-shaped shape.

![image](https://github.com/SiriusZhangyu/machine-learning-project/blob/patch-2/U-Net/unet.png)

  The right side of the network (green dotted line) is called expansive path in the paper. Similarly, it consists of four blocks. Before each block starts, it multiplies the size of the Feature Map by 2 by deconvolution, and reduces its number by half (the last layer is slightly different), then merges with the feature Map of the symmetrical compression path on the left. Because the size of the feature Map of the left compression path is different from that of the right extension path, U-Net cuts and expands the feature Map of the compression path through the feature Map of the compression path. Feature Maps of the same size of the spreading path are normalized.

  In this project, we plan to use Unit technology to extract the segment from the ships and train a sample DNN model to detective them. the main part of this program is from https://www.kaggle.com/hmendonca/u-net-model-with-submission#. And we simplify the code and change some parts based on our understanding to make it runs more effectively. 
  Here are several important parts.
  
  Process Data
  ------
  ![image](https://github.com/SiriusZhangyu/machine-learning-project/blob/patch-2/U-Net/parameters.png)
  
  Split Dataset And Combination Mask With Original Pictures
  -----
  Put the data into train and validation set.
  Set 1/5 of the dataset to be validation dataset.
  ![image](https://github.com/SiriusZhangyu/machine-learning-project/blob/patch-2/U-Net/splited%20data.png)
