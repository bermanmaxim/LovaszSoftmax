# The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks

Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko

ESAT-PSI, KU Leuven, belgium.

To appear in CVPR 2018. See [project page](http://bmax.im/LovaszSoftmax) and [arxiv paper](https://arxiv.org/abs/1705.08790).

Files included:
* **lovasz_losses.py**: PyTorch implementation of the Lovász hinge and Lovász-Softmax for the Jaccard index
* **utils.py**: Some Python utils
* **demo_binary.ipynb**: Jupyter notebook showcasing binary training of a linear model
* **demo_multiclass.ipynb**: Jupyter notebook showcasing multiclass training of a linear model

The binary `lovasz_hinge` expects real-valued scores (positive scores correspond to foreground pixels). 

The multiclass `lovasz_softmax` expect class probabilities (the maximum scoring category is predicted). First use a `Softmax` layer on the unnormalized scores.

*Experiments with Deeplab and ENet to be added later*
