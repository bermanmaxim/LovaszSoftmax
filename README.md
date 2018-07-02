# The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks

<img src="https://cdn.rawgit.com/bermanmaxim/bermanmaxim.github.io/5edecd41/single_LSimage.jpg" height="180">

Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko

ESAT-PSI, KU Leuven, Belgium.

Published in CVPR 2018. See [project page](http://bmax.im/LovaszSoftmax), [arxiv paper](https://arxiv.org/abs/1705.08790), [paper on CVF open access](http://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html).

## PyTorch implementation of the loss layer (*pytorch* folder)
**Files included:**
* **lovasz_losses.py**: Standalone PyTorch implementation of the Lovász hinge and Lovász-Softmax for the Jaccard index
* **demo_binary.ipynb**: Jupyter notebook showcasing binary training of a linear model
* **demo_multiclass.ipynb**: Jupyter notebook showcasing multiclass training of a linear model

The binary `lovasz_hinge` expects real-valued scores (positive scores correspond to foreground pixels). 

The multiclass `lovasz_softmax` expect class probabilities (the maximum scoring category is predicted). First use a `Softmax` layer on the unnormalized scores.

## TensorFlow implementation of the loss layer (*tensorflow* folder)
**Files included:**
* **lovasz_losses_tf.py**: Standalone TensorFlow implementation of the Lovász hinge and Lovász-Softmax for the Jaccard index
* **demo_binary_tf.ipynb**: Jupyter notebook showcasing the application of the binary loss
* **demo_multiclass_tf.ipynb**: Jupyter notebook showcasing the application of the multiclass loss

*Warning: the losses values and gradients have been tested to be the same as in PyTorch (see notebooks), however we have not used the TF implementation in a training setting.*

## Usage
See the demos for simple proofs of principle.

## FAQ
* How should I use the Lovász-Softmax loss?

The loss can be optimized on its own, but the optimal optimization hyperparameters (learning rates, momentum) might be different from the best ones for cross-entropy. As discussed in the paper, optimizing the dataset-mIoU (Pascal VOC measure) is dependent on the batch size and number of classes. Therefore you might have best results by optimizing with cross-entropy first and finetuning with our loss, or by combining the two losses. 

See for example how the work [*Land Cover Classification From Satellite Imagery With U-Net and Lovasz-Softmax Loss* by Alexander Rakhlin et al.](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Rakhlin_Land_Cover_Classification_CVPR_2018_paper.html) used our loss in the [CVPR 18 DeepGlobe challenge](http://deepglobe.org/).

## Citation
Please cite
```
@InProceedings{Berman_2018_CVPR,
author = {Berman, Maxim and Rannen Triki, Amal and Blaschko, Matthew B.},
title = {The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
