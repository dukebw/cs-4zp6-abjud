# TensorFlow HTTP Server

The idea of this module is to create a GPU ReST server, inspired by the [NVIDIA
GPU ReST Engine](https://github.com/NVIDIA/gpu-rest-engine) for Caffe.

Since the API would be totally based on HTTP requests, this would allow other
applications to use TensorFlow features without installing or setting up
anything related to TensorFlow. So for example, a mobile phone application
could make use of a server to do image classification by making use of this
ReST API.
