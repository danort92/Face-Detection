## FaceFinder - A Face Detection Model

![plot]([http://url/to/img.png](https://cdn.technologyreview.com/i/images/Face%20detection.png))

Face Detection algorithm that takes an image in input and provides the same image with bounding boxes around the detected faces. 
Creation from scratch, no use of pre-trained models.

The training phase is based on the following steps:
* _import Caltech images set;_
* _data augmentation through imgaug library's functions;_
* _use of a linear SVM classifier with optimized hyperparameters._

The testing phase is characterized by the following steps:
* _use of sliding window approach to identify the bounding boxes with confidence levels higher than a positive threshold;_
* _suppression of overlapping bounding boxes with lower confidence levels;_
* _load of the image URL link;_
* _if needed by the high number of false positives, run of an "hard negative mining" function to manually boost the classifier training (using bounding boxes with confidence levels lower than a negative threshold)._
* _plot of the image with bounding boxes._

It's possible to decide if:
1- a data augmentation should be performed on the initial dataset (may be useful for particular images with a high number of faces/non frontal faces);
2- the trained model should be saved for future uses (giving the path where the model needs to be saved);
3- a pre-saved model should be loaded (giving the path where the model is saved);
4- the hard negative mining should be performed (if the plotted image contains too many false positives).
