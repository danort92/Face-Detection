# **FaceFinder - A Face Detection Model**



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




## Code details
In this notebook I am going to develop a Face Detection Model given a photo as input. I won't use pre-trained models, but will create my own from scratch.
I will use imgaug library to manage images and scikit-learn functions
```
!pip install opencv-python
!pip install imgaug

import os
from PIL import Image
import glob
import collections
import pickle
import time

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmenters import Fliplr, Flipud, Affine
from itertools import chain
from skimage import data, color, feature
import skimage.data
from skimage.transform import resize

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV

#say no to warnings!
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
```
### IMPORT DATA

I will train the model using positive and negative images from Caltech dataset. It contains 6713 positive (w/ faces) and 275 negative (w/o faces) images.

```
!git clone https://github.com/danort92/Face-Detection.git
pos_path="/Face-Detection/sets/train_positive_set/Caltech_CropFaces"
neg_path="/Face-Detection/sets/train_negative_set"
pos_set=create_img_array(glob.glob (os.getcwd() + pos_path + "/*.jpg"))
neg_set=create_img_array(glob.glob (os.getcwd() + neg_path + "/*.jpg"))
```

### DATA AUGMENTATION

Dataset is augmented with slightly modified images. Negative pics are horizontally and vertically flipped, rotated and cropped. Positive pics are just mirrored (applying the same modifies as done for the negative set I've seen to be linked to many false positives, so I preferred only the less invasive modification)

- original script found on https://imgaug.readthedocs.io/ and slightly modified in order to perform data augmentation
```
def img_neg_augmentation(img_set,neg_k=1):
    """
    FUNC: Data augmentation through different techniques (horiziontal flip, vertical flip, rotation and shear)
          Images are randomly chosen through random choice of index
    ARGS:
        - img_set: (N,2) ndarray; N is the number of negative images and each row represents width and height.
        - neg_k:  constant, multiplier of images randomly chosen from the original set
    RETURN:
        - aug_set: (M,2) ndarray; M is the number of negative images (M>N) and each row represents width and height
    """

    n_images=np.array(img_set).shape[0]
    aug_set=[]
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    for i in range(1,int(neg_k/4*n_images)):
        r=np.random.choice(n_images,1)[0]

        hflip= iaa.Fliplr(p=0.6)
        aug_hor_image= hflip.augment_image(img_set[r])
        aug_set.append(aug_hor_image)

        vflip= iaa.Flipud(p=0.2)
        aug_ver_image= vflip.augment_image(img_set[r])
        aug_set.append(aug_ver_image)

        aff = sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-30, 30),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL))
        aug_aff_image= aff.augment_image(img_set[r])
        aug_set.append(aug_aff_image)

        crop=sometimes(iaa.Crop(percent=(0, 0.1)))
        aug_crop_image= crop.augment_image(img_set[r])
        aug_set.append(aug_crop_image)

    return aug_set
```    
```
def img_pos_augmentation(img_set,pos_k):
    """
    FUNC: Data augmentation only through horiziontal flip
          (applying the same techniques used for negative set there would be many more false positives)
          Images are randomly chosen through random choice of index
    ARGS:
        - img_set: (N,2) ndarray; N is the number of positive images and each row represents width and height.
        - pos_k:  constant, multiplier of images randomly chosen from the original set
    RETURN:
        - aug_set: (M,2) ndarray; M is the number of positive images (M>N) and each row represents width and height
    """

    n_images=np.array(img_set).shape[0]
    aug_set=[]
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    for i in range(1,int(pos_k*n_images)):
        r=np.random.choice(n_images,1)[0]

        hflip= iaa.Fliplr(p=0.6)
        aug_hor_image= hflip.augment_image(img_set[r])
        aug_set.append(aug_hor_image)

    return aug_set
```
### SVM MODEL

A linear Support Vector Machine model is used. 

- Hyperparameters are optimized through the script found on https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV. It iterates to find the best estimators for the binary classification model GridSearchCV

```
def print_dataframe(filtered_cv_results):
    """Pretty print for filtered dataframe"""
    for mean_precision, std_precision, mean_recall, std_recall, params in zip(
        filtered_cv_results["mean_test_precision"],
        filtered_cv_results["std_test_precision"],
        filtered_cv_results["mean_test_recall"],
        filtered_cv_results["std_test_recall"],
        filtered_cv_results["params"],
    ):
        print(
            f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
            f" for {params}"
        )
    print()


def refit_strategy(cv_results):
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter-out all results below a precision threshold
    of 0.97, rank the remaining by recall and keep all models with one standard
    deviation of the best by recall. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    # print the info about the grid-search for the different scores
    precision_threshold = 0.97

    cv_results_ = pd.DataFrame(cv_results)
    print("All grid-search results:")
    print_dataframe(cv_results_)

    # Filter-out all results below the threshold
    high_precision_cv_results = cv_results_[
        cv_results_["mean_test_precision"] > precision_threshold
    ]

    print(f"Models with a precision higher than {precision_threshold}:")
    print_dataframe(high_precision_cv_results)

    high_precision_cv_results = high_precision_cv_results[
        [
            "mean_score_time",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "rank_test_recall",
            "rank_test_precision",
            "params",
        ]
    ]

    # Select the most performant models in terms of recall
    # (within 1 sigma from the best)
    best_recall_std = high_precision_cv_results["mean_test_recall"].std()
    best_recall = high_precision_cv_results["mean_test_recall"].max()
    best_recall_threshold = best_recall - best_recall_std

    high_recall_cv_results = high_precision_cv_results[
        high_precision_cv_results["mean_test_recall"] > best_recall_threshold
    ]
    print(
        "Out of the previously selected high precision models, we keep all the\n"
        "the models within one standard deviation of the highest recall model:"
    )
    print_dataframe(high_recall_cv_results)

    # From the best candidates, select the fastest model to predict
    fastest_top_recall_high_precision_index = high_recall_cv_results[
        "mean_score_time"
    ].idxmin()

    print(
        "\nThe selected final model is the fastest to predict out of the previously\n"
        "selected subset of best models based on precision and recall.\n"
        "Its scoring time is:\n\n"
        f"{high_recall_cv_results.loc[fastest_top_recall_high_precision_index]}"
    )

    return fastest_top_recall_high_precision_index
```
- Transformation of images in HOG vectors is performed, because HOG relies on the property of objects within an image to possess the distribution of intensity gradients or edge directions.
              Gradients are calculated within an image per patch. A patch is considered as a pixel grid in which gradients
              are constituted from the magnitude and direction of change in the intensities of the pixel within the patch.
```
def train_classifier(pos_set,neg_set,pixel_per_cell=(6,6)):

    """
    FUNC: 1 - The train features and labels are created from predefined positive and negative sets.
          2 - The images are transformed in vectors through HOG feature from skimage library
          3 - A Linear Support Vector Classification model is then trained in order to optimize precision and recall through cross-validation.
    ARGS:
        - pos_set: (N,2) ndarray; N is the number of positive images and each row is width and height.
        - neg_set: (N,2) ndarray; N is the number of negative images and each row is width and height.
    RETURN:
        - model: the model fitted with best found estimators
    """

    X_train = np.array([feature.hog(im, pixels_per_cell=pixel_per_cell)
                        for im in chain(pos_set,
                                        neg_set)])
    y_train = np.zeros(X_train.shape[0])
    y_train[:np.shape(pos_set)[0]] = 1

    scoring = ['precision','recall']
    grid = GridSearchCV(LinearSVC(random_state=1),
                        {"C": [0.001,0.0025,0.005,0.01,0.025,0.05]},
                        scoring=scoring,
                        verbose=2,
                       refit=refit_strategy)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    model.fit(X_train, y_train)

    return model
```
### TESTING

During the testing phase a sliding window approach is used to slide on the image given a fixed step and find the patches' confidence. The patches with confidence higher than a positive threshold are considered as bounding boxes with enough probability to contain a face. Overlapping bounding boxes with lower confidences are then suppressed. Patches with confidence lower than a negative threshold, instead, have a high probability to not contain any faces and are saved for a subsequent hard mining phase where they aer used to manually boost the model
```
def non_max_supr_bbox(bboxes, confidences, img_size):

    """
    FUNC: high confidence detections suppress all overlapping detections
        (including detections at other scales). Detections can partially
        overlap, but the center of one detection can not be within another
        detection.
    ARGS:
        - bboxes: (N,4) ndarray; N is the number of non-overlapping detections,
                  and each row is [x_min, y_min, x_max, y_max].
        - confidences: (N,1) ndarray; Confidence of each detection after final
                  cascade node.
        - img_size: (2,) list; width and height of the image.
    RETURN:
        - is_valid_bbox: (N,1) bool ndarray; indicating valid bounding boxes.
    """

    #Truncate bounding boxes to image dimensions
    x_out_of_bounds=bboxes[:,2] > img_size[1]  #xmax greater than x dimension
    y_out_of_bounds=bboxes[:,3] > img_size[0]  #ymax greater than y dimension
    bboxes[x_out_of_bounds,2]=img_size[1]
    bboxes[y_out_of_bounds,3]=img_size[0]

    num_detections=confidences.shape[0]

    #higher confidence detections get priority
    ind=np.argsort(-confidences, axis=0).ravel()
    bboxes=bboxes[ind,:]

    #indicator for whether each bbox will be accepted or suppressed
    is_valid_bbox=np.zeros((num_detections,1),dtype=np.bool)
    for i in range(num_detections):
        cur_bb=bboxes[i,:]
        cur_bb_is_valid=True

        for j in np.where(is_valid_bbox)[0]:
            prev_bb=bboxes[j,:]
            bi=[max(cur_bb[0], prev_bb[0]),
                max(cur_bb[1], prev_bb[1]),
                min(cur_bb[2], prev_bb[2]),
                min(cur_bb[3], prev_bb[3])]
            iw=bi[2]-bi[0]+1
            ih=bi[3]-bi[1]+1

            if iw>0 and ih>0:
                #compute overlap as area of intersection / area of union
                ua=(cur_bb[2] - cur_bb[0] + 1) * (cur_bb[3] - cur_bb[1] + 1) + \
                     (prev_bb[2] - prev_bb[0] + 1) * (prev_bb[3] - prev_bb[1] + 1) - \
                     iw * ih
                ov = iw * ih / ua

                #if the less confident detection overlaps too much with the previous detection
                if ov>0.2:
                    cur_bb_is_valid=False

                center_coord=[(cur_bb[0] + cur_bb[2]) / 2, (cur_bb[1] + cur_bb[3]) / 2]
                if (center_coord[0] > prev_bb[0]) and (center_coord[0] < prev_bb[2]) and \
                        (center_coord[1] > prev_bb[1]) and (center_coord[1] < prev_bb[3]):
                    cur_bb_is_valid=False

        is_valid_bbox[i]=cur_bb_is_valid

    #This statement returns the logical array 'is_valid_bbox' back to the order
    #of the input bboxes and confidences
    reverse_map=np.zeros((num_detections,), dtype=np.int)
    reverse_map[ind]=np.arange(num_detections)
    is_valid_bbox=is_valid_bbox[reverse_map,:]

    return is_valid_bbox
```
```
def sliding_window(path, patch_size=(36,36), pixel_per_cell=(6,6), step=3, p_threshold=1, n_threshold=-0.5, downsample=0.8, verbose=False):

    """
    FUNC: a patch of predefined size slides on the image with a specific step.
          1 - For each step i) the image is transformed through Histogram Of Gradients (HOG) function
              and ii) a confidence level is calculated based on the trained model.
          2 - If the conficence is higher than a predefined threshold the patch is kept because it has
              a relatively high probability to contain a face. A negative threshold is also set to keep patch with a low confidence
              (to eventually perform a negative hard mining, including those pathces in the train set manually boost the model)
          3 - The previous points are repeated at different scales (there could be faces with different dimensions inside the image)
    ARGS:
        - path: string - the path of the image
        - patch_size: (2,) tuple; pixel dimensions of the patch (Width,Height)
        - step: constant; pixel step between two condecutive patches and defining the sliding window mechanism
        - p_threshold: constant; threshold for high confidence patches (default=1, the lower the higher the risk of false positives)
        - n_threshold: constant; threshold for low confidence patches (default=-0.5, the lower )
        - downsample: constant; the parameter defines the quantity the image is scaled after each loop
        - verbose: boolean; shows some interesting parameters (default=False)
    RETURN:
        - plt.show(): plot of the image together with the detected bounding boxes
    """

    img_name=os.listdir(path)

    bboxes = np.zeros([0, 4])
    confidences = np.zeros([0, 1])

    hard_neg_set=[]
    k=0

    img = cv2.imread(os.path.join(path, img_name[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #process starts from an initial scale defined by this ratio
    #because i) there would be much more false positives and ii) no camera gets to detected very small faces in an image
    scale_factor=4*patch_size[0]/min(img.shape[0],img.shape[1])

    W, H = (int(s) for s in patch_size)
    print("------------------------------------------")
    print("Detection STARTED...\n")

    while min(img.shape[0],img.shape[1])*scale_factor >= patch_size[0]:

        img_re=resize(img, (int(img.shape[0]*scale_factor), int(img.shape[1]*scale_factor)))


        for i in range(0, img_re.shape[0] - W, step):
            for j in range(0, img_re.shape[1] - H, step):

                patch = img_re[i:i + W, j:j + H]
                patch_hog = feature.hog(patch, pixels_per_cell=pixel_per_cell)

                confidence = model.decision_function(patch_hog.reshape(1, -1))
                if confidence>=p_threshold:

                    x_min = j/scale_factor
                    y_min = i/scale_factor
                    x_max = (j+W)/scale_factor
                    y_max = (i+H)/scale_factor

                    bboxes = np.concatenate((bboxes, [[x_min, y_min, x_max, y_max]]), 0)
                    confidences = np.concatenate((confidences, [confidence]), 0)

                elif confidence<n_threshold:
                    hard_neg_set.append(patch)
        if verbose:
          print(f"{k+1}° loop completed --> Scale factor: {scale_factor*100:.2f}%")

        k+=1
        scale_factor = scale_factor * downsample

    #added to remove bounding boxes overlapping each others, with preference to higher confidence
    is_maximum = non_max_supr_bbox(bboxes, confidences, img.shape)

    bboxes = bboxes[is_maximum[:, 0], :]
    confidences = confidences[is_maximum[:, 0], :]
    print("\nDetection COMPLETED!")
    print("------------------------------------------\n")
    print(f"Bounding boxes found: {len(bboxes)}\n")
    return bboxes, confidences, hard_neg_set
```

### HARD MINING

This functiondefines the hard mining method, where patches with a not high enough confidence are added to the augmented negative set
def hard_negative_mining(hard_neg_set,neg_set,pos_set, n_hard_mined=5000):
```
    """
    FUNC: takes a defined number of npatches with confidences lower than a defined threshold, add them to the negative images set
          and re-train the model in order to reduce false positive detections
          *** Use only in presence of many false positives, since it's a bit time consuming!
    ARGS:
        - hard_neg_set: (M,2) ndarray; M is the number of patches with confidence lower than a negative threshold
                        and each row is width and height
        - neg_set: (N,2) ndarray; N is the number of negative images and each row is width and height
        - pos_set: (P,2) ndarray; P is the number of positive images and each row is width and height
        - n_hard_mined: constant; number of patches used for hard mining

    RETURN:

        - neg_set: (K,2) ndarray; K=N+n*M is the number of negative images plus slected nageative patches
                   and each row is width and height
    """

    n_hard_neg=np.array(hard_neg_set).shape[0]
    hard_neg_subset=[]
    i=0

    while i < n_hard_mined:
        k=np.random.choice(n_hard_neg,1)[0]
        hard_neg_subset.append(hard_neg_set[k])
        i+=1

    neg_set=np.concatenate((neg_set, hard_neg_subset))
    print("\nHard negative mining STARTED!")
    print("----------------------------------------------------------------------------------------------------------\n")

    return neg_set
```



### RUNNING CODE

Before providing the URL link of the image you want to detect faces, you can decide to:
- run the data augmentation on the train set (it may help in improving detection)
- save the trained model for future use and choose location where it will be saved
- load an already trained model from a specified location
  
You can also decide to perform an hard negative mining (it may improve the detection reducing false positives).

