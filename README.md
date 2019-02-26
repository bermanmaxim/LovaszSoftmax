# The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks

<img src="https://cdn.rawgit.com/bermanmaxim/bermanmaxim.github.io/5edecd41/single_LSimage.jpg" height="180">

Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko

ESAT-PSI, KU Leuven, Belgium.

Published in CVPR 2018. See [project page](http://bmax.im/LovaszSoftmax), [arxiv paper](https://arxiv.org/abs/1705.08790), [paper on CVF open access](http://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html).

## PyTorch implementation of the loss layer (*pytorch* folder)
**Files included:**
* **lovasz_losses.py**: Standalone PyTorch implementation of the Lovász hinge and Lovász-Softmax for the Jaccard index
* **demo_binary.ipynb**: Jupyter notebook showcasing binary training of a linear model, with the Lovász Hinge and with the Lovász-Sigmoid.
* **demo_multiclass.ipynb**: Jupyter notebook showcasing multiclass training of a linear model with the Lovász-Softmax

The binary `lovasz_hinge` expects real-valued scores (positive scores correspond to foreground pixels). 

The multiclass `lovasz_softmax` expect class probabilities (the maximum scoring category is predicted). First use a `Softmax` layer on the unnormalized scores.

## TensorFlow implementation of the loss layer (*tensorflow* folder)
**Files included:**
* **lovasz_losses_tf.py**: Standalone TensorFlow implementation of the Lovász hinge and Lovász-Softmax for the Jaccard index
* **demo_binary_tf.ipynb**: Jupyter notebook showcasing binary training of a linear model, with the Lovász Hinge and with the Lovász-Sigmoid.
* **demo_multiclass_tf.ipynb**: Jupyter notebook showcasing the application of the multiclass loss with the Lovász-Softmax

*Warning: the losses values and gradients have been tested to be the same as in PyTorch (see notebooks), however we have not used the TF implementation in a training setting.*

## Usage
See the demos for simple proofs of principle.

## FAQ
* How should I use the Lovász-Softmax loss?

The loss can be optimized on its own, but the optimal optimization hyperparameters (learning rates, momentum) might be different from the best ones for cross-entropy. As discussed in the paper, optimizing the dataset-mIoU (Pascal VOC measure) is dependent on the batch size and number of classes. Therefore you might have best results by optimizing with cross-entropy first and finetuning with our loss, or by combining the two losses. 

See for example how the work [*Land Cover Classification From Satellite Imagery With U-Net and Lovasz-Softmax Loss* by Alexander Rakhlin et al.](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Rakhlin_Land_Cover_Classification_CVPR_2018_paper.html) used our loss in the [CVPR 18 DeepGlobe challenge](http://deepglobe.org/).

* Inference in Tensorflow is very slow...

Compiling from Tensorflow master (or using a future distribution that includes commit [tensorflow/tensorflow@73e3215](https://github.com/tensorflow/tensorflow/commit/73e3215c3a2edadbf9111cca44ab3d5ca146c327)) should solve this problem; see [issue #6](https://github.com/bermanmaxim/LovaszSoftmax/issues/6).

## Citation
Please cite
```
@inproceedings{berman2018lovasz,
  title={The Lov{\'a}sz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks},
  author={Berman, Maxim and Rannen Triki, Amal and Blaschko, Matthew B},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4413--4421},
  year={2018}
}
```
