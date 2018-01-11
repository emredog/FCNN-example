# FCNN Example

This was an exercise for a job application (of which I will not disclose the company name).

The goal was overfit to the given single image, a large aerial image along with ground truth, so that we can detect houses.

The network architecture was given, so no flexibility there. 

This repo contains PyTorch code and other material to decribe my approach to the given problem.

*Note* that custom cross entropy function is in fact unnecessary, and will be removed in the next version. 

## Repo Structure
* `images` folder contains sample image and its ground truth
* `plots` folder contains loss and score plots of a trained model
* `predicted_images` folder contains images marked with the resulting prediction of a trained model
* `weights` folder contains model weights, where a model may usually have multiple checkpoints
* `FCNN.py` is the fully convolutional neural net model as defined in the exercise. Upscaling is constant and nearest neighbor interpolation is used by default.
* `FCNN2.py` is almost the same model as defined `FCNN.py`, **but** with a learnable upscaling layer, i.e. *Transposed Convolution*, following [1].
* `train.py` is the script to train the network with given parameters for given epochs. It also:
  * Saves a plot containing training loss per iterations
  * Saves a plot containing model scores (see below for details) per epochs
  * Saves sets of model weights with best scores, as well as the last set of weights.
* `predict.py` is the simplest script of all, simply predicting the data provided by given *dataloader* using given *trained model* 
* `util.py` is a collection of helper functions and some constants to make life easier. Basically, it has functions for
  * Loading the image (and dividing it into patches),
  * Analyzing the patches (elininating blank patches, or patches with low information) as well as other image operations (data augmentation, preprocessing etc.),
  * Calculating scores for given predictions,
  * Saving the segmentation result upon the image to have a nice image at the end.
* `main.py` is the script that is used in the experiments, to try out different hyperparameters in a loop. It also demonstrates the intended usage of the `train.py` and `predict.py`.
* `CrossEntropyLoss2d.py` is a custom loss function obtained from [here](https://discuss.pytorch.org/t/about-segmentation-loss-function/2906/6). Although the original [Cross Entropy Loss](http://pytorch.org/docs/0.3.0/nn.html#crossentropyloss) of PyTorch supports tensors of any size, ~my experiments showed that the custom loss allows better performance~ No, they both yield the exact same result.

## Performance Metrics
Initially, only "classification accuracy" (or "accuracy" for short) was used, since the problem was implemented a binary classification task. Shortly after, I've realized that the data is highly unbalanced (~91% background vs. ~9% houses), e.g. a result with no detection at all would result in 91% accuracy! (But I have kept this metric nonetheless, to compare with earlier attempts)

Therefore, if we frame the problem as a "house detection" problem, we can use metrics like **Precision**, **Recall** and **F1 Score**. These would enable us to properly evaluate the model performance, and to compare hyperparameters.  

I have also considered IoU metric, but decided that would be counterintuitive for semantic segmentation and too much of a hassle for a 1 week exercise.

## Unbalanced data --> Class Weights in Loss Function
As mentioned earlier, sample data is highly unbalanced, where 91% of the pixels are background whereas only 9% are houses. This becomes a huge burden during training, because loss from the houses becomes too insignificant compared to loss from background pixels, which causes the optimizer to be contented in situations such as "very few detections". 

To address this issue, providing class weights into the Loss function is a good option under this circumstances (where we can't get more data). This is in fact forces the optimizer to find the weights that yields low loss from background pixels and low (despite amplified!) loss from house pixels.

I thought that weights **1 vs. 10** would work best theoretically, but **1 vs 6** and **1 vs 8** turned out to be better empirically.

## Training & Test
This section provides a brief description about the training and test procedures.

### Image patches
The requirements of the exercise clearly states that the images patches feed into the network must not be larger than 256x256 pixels. Moreover, I wanted to experiment with other image sizes as well.

To this end, data loader functions in `util.py` take `W` as argument to calculate how to divide the large sample image into `WxW` patches.

> **Assumption** For convenience, image patches are always square, hence `WxW` and not `WxH`. Extending existing code to handle rectangular inputs is trivial, but I believe that is very unusual in the literature.

Image patches are created differently for Training and Test stages:
* **Training**: Sample image is divided into `WxW` patches. Zero paddings around the sample images was used to avoid remainders at the boundaries.
  1. Patches with more than 1/2 blank are discarded **if** they do not contain houses. 
  2. For **augmented** case, different **strides** are used to crop the images, to provide overlapping patches. In my experiments, using `W/2` strides provided 8% improvement in F1 Score. For this exercise, I decided not to apply further data augmentation (random crops, random flips/rotations, color jittering etc.), since *generalization* was not a concern. 
  
* **Test**: At test time, since we shouldn't know where the houses are, the sample is image is simply zero padded and divided into `WxW` patches.

Please note that, both training and test patches were normalized with respect to a *mean image* computed from training set. (see `images/mean.npy`)

### Hyperparameters and other choices
During searching for hyperparameters, the random seed was fixed. After long training hours, I found that following hyperparameters works the best for the given.
* **Image size (W)**: 128
* **Batch size (N)**: 1
* **Upscale method (u)**: Learnable (`FCNN2`) with transposed convolutions
* **Learning rate (lr)**: 1e-4
  * with **decay**: 0.5 at every 40 **step**
* **Optimizer**: Following [1], SGD optimizer was used. I found that **momentum**:0.9 works best.
* **Weight initialization**: He et al.[2] initialization with normal distribution for all learnable weights in the network.
* **Regularization**: L2 regularization with strength 5e-3.
* **Class weights for loss**: 1.0 for background, 6.0 for house

More experiments (although not presented nicely) can be found [here](https://docs.google.com/spreadsheets/d/1oTYpziq4KZalAB9y6hmJ9XI8HsC-icptbH_BWsMO2sM/edit?usp=sharing).

### Results
Results for the model #236, which is trained 200 epochs with the indicated hyperparams above.

| Precision | Recall | F1 Score | Accuracy |
|:---------:|:------:|:--------:|:--------:|
| 76.80     |	93.56	 | 84.35    |97.55     |

**Training Loss over iterations:**
<div style="text-align:center"><img src ="https://github.com/emredog/FCNN-example/raw/master/plots/234_losses.png" /></div>

**Scores over epochs:**
<div style="text-align:center"><img src ="https://github.com/emredog/FCNN-example/raw/master/plots/234_acc.png" /></div>

**Qualitative result:**
![Predictions](https://github.com/emredog/FCNN-example/raw/master/predicted_images/234.png)


## References
1. Long, J., Shelhamer, E., & Darrell, T. Fully convolutional networks for semantic segmentation. CVPR, 2015.
2. He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification.", ICCV. 2015.
