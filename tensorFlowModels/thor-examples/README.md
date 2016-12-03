# Discussion of TensorFlow Example Code

## Data Augmentation

We need to be able to increase the number of examples in the dataset by
augmenting data, e.g. by rotating images. This way the model has more examples
to learn from when it comes to finding all the joints. Labels must be augmented
in the same way as the images.

## Scaling Labels

To make the labels invariant to fixed-aspect-ratio scaling, coordinates should
be in the range [0, 1.0] for each dimension. E.g. for a 400x400 image (0.5,
0.5) is a valid co-ordinate, but (200, 200) is not and should be re-scaled.

## DeepPose and Using Pre-trained Architectures

The DeepPose paper does pose estimation in stages, where each stage looks at a
bounding box of 220x220 pixels with increased resolution. Each stage has its
own parameters. The first stage is the whole image, and the last stage is the
minimal square for each joint. This can be adjusted, and results compared.

The architecture used in DeepPose was AlexNet, and you can swap in any widely
available network such as VGG-16.

[TFSlim](https://github.com/tensorflow/models/tree/master/slim) can be used for
pre-trained architectures.

## First Step

Playing with the TF image API and seeing whether the labels can be augmented
correctly after the image is augmented would be a good first step.

[PIL (Pillow)](http://effbot.org/imagingbook/imagedraw.htm) can be used to draw
on the image.
