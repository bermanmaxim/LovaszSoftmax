# The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks

Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko

ESAT-PSI, KU Leuven, Belgium.

To appear in CVPR 2018. See [project page](http://bmax.im/LovaszSoftmax) and [arxiv paper](https://arxiv.org/abs/1705.08790).

## PyTorch implementation of the loss layer
**Files included:**
* **lovasz_losses.py**: PyTorch implementation of the Lovász hinge and Lovász-Softmax for the Jaccard index
* **utils.py**: Some Python utils
* **demo_binary.ipynb**: Jupyter notebook showcasing binary training of a linear model
* **demo_multiclass.ipynb**: Jupyter notebook showcasing multiclass training of a linear model

The binary `lovasz_hinge` expects real-valued scores (positive scores correspond to foreground pixels). 

The multiclass `lovasz_softmax` expect class probabilities (the maximum scoring category is predicted). First use a `Softmax` layer on the unnormalized scores.

## TensorFlow implementation of the loss layer
**Files included:**
* **lovasz_losses_tf.py**: TensorFlow implementation of the Lovász hinge and Lovász-Softmax for the Jaccard index
* **utils.py**: Some Python utils
* **tfutils.py**: Some TensorFlow utils
* **demo_binary_tf.ipynb**: Jupyter notebook showcasing the application of the binary loss
* **demo_multiclass_tf.ipynb**: Jupyter notebook showcasing the application of the multiclass loss

*Warning: the losses values and gradients have been tested to be the same as in PyTorch (see notebooks), however we have not used the TF implementation in a training setting.*

## Experiments
To be added later. See the demos for simple proofs of principle.
