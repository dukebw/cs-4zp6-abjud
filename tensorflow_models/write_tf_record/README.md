# `write_tf_record` Related Information

## Writing TensorFlow Records

It is possible to write TFRecords using the `tf.python_io.TFRecordWriter`
class, which can be used in Python `with` blocks, and whose purpose is to write
records to a TFRecords file.

A `tf.python_io.TFRecordWriter` object is initialized with the filename of the
output file. We want to call the `tf.python_io.TFRecordWriter.write` method on
(TODO(brendan): refer to Features protobuf format)

### Some Profiling

Using only a single Python thread, the following performance results were
obtained from `write_tf_record` (note that all times are approximate):

#### Using TensorFlow `QueueRunner` to read files

Note that this seems to start two threads, but decoding is done in serial via
the `tf.decode_jpeg` call.

512 images: 43s

1024 images: 105s

2048 images: 208s

#### Using Python to read files (i.e. `open(file, 'rb')`):

##### With one thread:

2048 images: 140s

##### With two threads:

512 images: 34s

1024 images: 56s

2048 images: 111s

##### With three threads:

2048 images: 90s

##### With four threads:

512 images: 24s

1024 images: 47s

2048 images: 90s

The filename list passed to `write_tf_record` was re-shuffled each run.

The results suggest that there is some overhead to the `QueueRunner` method,
probably since we decode only one image at a time before processing it from
Python.

There was also some speed-up from multi-threading up to three threads, which
makes sense since the machine has four hardware threads in total (two cores,
each with two hardware threads), and at the time of experiment one of those
hardware threads was used to run at 100% usage running a MATLAB program.

### Reference to [Inception](https://github.com/tensorflow/models/tree/master/inception)

#### TF Record Writing Code

Referencing `data/build_imagenet_data.py`, the functions `_build_synset_lookup`
and `_build_bounding_box_lookup` map synsets (e.g. n02119247) to labels (e.g.
black fox), and bounding boxes, respectively.

Then, `_find_image_files` returns a set of all image filenames, synsets and
labels, where the image filenames have been shuffled so that the order of
examples in each TFRecord is random with respect to the example's label.

`_process_image_files` splits the filenames into equal ranges, one for each
thread in `FLAGS.num_threads`. Each thread executes the
`_process_image_files_batch` function.

All threads use the same `coder` object, an instance of `ImageCoder`, which is
a wrapper to pass in a TF session used to run all the image decoding.

The training data are split up into shards, where a shard refers to a distinct
TFRecord file. There are 1024 training shards and 128 validation shards by
default. The shards are processed in per-thread batches. So for instance if
there were 4 threads and 128 shards, then the first thread would process shards
\[0, 32)

Within each thread, the images are decoded and raw image data, along with
labels, is written as protobuf to the TFRecord corresponding to each shard.

#### Training

##### Input Pipeline

Images and labels are produced by the `distorted_inputs` function.

Using `string_input_producer`, a queue (`filename_queue`) containing the
filenames of all of the TFRecord files, and capacity 16, is added to the graph
(along with a `QueueRunner` that runs the subgraph to fill the queue).

A set of reader threads (by default four, again on the CPU) are used to read
TFRecord examples from disk. Each reader reads TFRecords whose filenames are
dequeued from `filename_queue`, and enqueues those TFRecords in a
`RandomShuffleQueue` called `examples_queue`. `examples_queue` is dequeued (in
random order due to the use of `RandomShuffleQueue` rather than `FIFOQueue`) to
produce serialized protobuf examples for parsing.

`examples_queue` is given a capacity of 16\*1024, where 16 is the default
value of `FLAGS.input_queue_memory_factor`. This is based on an estimate of
about 1024 examples per shard, and 1MB per image. However the 1MB per image
estimate seems to be incorrect, since it is based on 299\*299\*3\*4 bytes, and
most of those bytes do not need the factor of 4 (since the image is stored in a
`bytes` array). Furthermore, due to the JPEG compression, example size appears
to be closer to 100KB than 1MB, based on the size of any given TFRecord shard
(approximately 135MB).

The purpose of multiplying by a factor of 16 is to pass `RandomShuffleQueue` a
minimum capacity, to maximize the level of mixing of the examples.

A set of CPU preprocessing threads equal to
`FLAGS.num_preprocess_threads*FLAGS.num_gpus` are used (by default four threads
per GPU) to parse and process example protobufs from `examples_queue`.

The serialized examples are parsed using a feature map, which is a Python
dictionary similar to the dictionary used to write the TF record. This feature
map is passed to `tf.parse_single_example`.

Image preprocessing involves first decoding the JPEG image and converting it to
float32.

Then, the image's human-annotated bounding box is distorted, with certain
constraints such as at least one tenth of the object must still be covered. The
image is then cropped to this distorted bounding box.

The cropped image is then resized to `FLAGS.image_size*FLAGS.image_size`, which
defaults to 299x299.

The image is then randomly flipped horizontally using
`tf.image.random_flip_left_right`, and colours are distorted with random
brightness, saturation, hue and contrast.

Images are scaled such that their colour values lie in \[-1, 1\] instead of
\[0, 1) as they are after conversion to float32.

By passing a list of length `num_threads` containing preprocessed image tensors
to `tf.train.batch_join`, `num_threads` threads are started, all enqueueing
preprocessed images originating (as serialized examples) from `examples_queue`.
The preprocessing is slightly different per thread, e.g. by applying colour
distortion operations in a different order.

In summary, there are three queues: `filename_queue`, containing the TFRecord
filenames, which feeds `examples_queue` with serialized example protobufs.
Finally, serialized examples are dequeued from `examples_queue`, preprocessed
in parallel by `num_preprocess_threads` and the result is enqueued into the
queue created by `batch_join`. A dequeue operation from the `batch_join` queue
is what is returned from `image_processing.distorted_inputs`. What is dequeued
is a batch of size `FLAGS.batch_size` containing a set of, for example, 32
images in the case of `images` or 32 labels in the case of `labels`.

##### Inference and Loss

Each batch of images and labels is split evenly across the GPUs using
`tf.split`.

Loss is computed per GPU, i.e. for each batch split, in the `_tower_loss`
function.

TODO(brendan): more explanation about the Inception architecture, auxiliary
loss, etc.

A tuple (logits, aux_logits) of logits, or unnormalized log-probability, values
is returned from the Inception inference graph (see
[the paper](https://arxiv.org/abs/1512.00567)) function `inception.inference`.
The list's shape is \[`batch_size`, `num_classes + 1`\], where e.g. for
ImageNet `num_classes` is 1000.

The cross-entropy loss is computed on both the main logits and the auxiliary
logits, where the auxiliary logits' loss is decreased by a factor of 0.4.
Rather than return any value, the losses are added to the default graph's
`LOSSES_COLLECTION` collection.

This loss, computed by comparing the model's predictions (from the logits) with
the expected values (labels) is added to a regularization loss calculated at
the same time over all of the weights in the network as the L2 norm of the
weight, multiplied by a `weight_decay` factor of 0.00004.

The regularization losses are pulled from `tf.GraphKeys.REGULARIZATION_LOSSES`,
where they were added by `tf.get_variable` during weights variable creation.

The resultant sum of loss, auxiliary loss and regularization losses is
`total_loss`.

A moving average of each loss from the main loss, auxiliary loss and total loss
is kept using `tf.train.ExponentialMovingAverage`.

Moving averages are computed as follows,

```python
    shadow_variable = decay * shadow_variable + (1 - decay) * variable
```

where a `shadow_variable` is tracking each of the losses. This moving average
is tracked in a summary, which can be viewed in TensorBoard.

A `global_step` variable is used to keep track of the number of steps, where
each step is one "split batch" being processed on one GPU, i.e. the number of
steps is the number of batches processed multiplied by the number of GPUs.

The learning rate defaults to 0.1, and is decayed (using `global_step`) by a
factor of `decay_rate` every N epochs, where an epoch is the entire dataset and
N is `FLAGS.num_epochs_per_decay`, which defaults to 30. A decay is computed at
each step as follows.

```python
decayed_learning_rate = learning_rate *
		  decay_rate ^ (global_step / decay_steps)
```

An RMSProp optimizer is used to compute gradients on the `total_loss`
returned from `_tower_loss` for each tower, which are averaged at a
synchronization point between towers.

Minimization is then applied to the averaged gradients using `apply_gradients`.
This is the point where `global_step` is incremented.

The averages of all trainable variables are tracked, again with
`tf.train.ExponentialMovingAverage`.

`tf.train.Saver` is used to save a checkpoint of all variables every 5000th
training step.

`allow_soft_placement=True` is passed in a `tf.ConfigProto` to the `tf.Session`
constructor, in order to allow towers to be built on the GPU.

The usual sequence of initializing all variables via
`sess.run(tf.initialize_all_variables())`, starting the `QueueRunner`s with
`tf.train.start_queue_runners`, and running the loss and minimization
operations for every step is followed.

### <a name="protobuf-heading"></a>Google Protobuf Format

#### Introduction

Google Protocol Buffers (a.k.a. protobuf) are a binary format of data storage
that can be used in various applications, e.g. in TensorFlow for storing
training and test datasets.

The advantages of using protobufs are in performance and data size: compared
with XML, protobufs are 3 to 10 times smaller, and parsing of protobufs is 20
to 100 times faster.

The main disadvantage of protobufs is that, since it is a binary format, it is
not human readable, and a `.proto` file is needed to describe the structure of
the data so that it can be de-serialized (i.e. converted from binary format
into the object structure of a programming language such as Python).

To read more about the motivation and history behind protobufs, see the
[Protocol Buffers Overview](https://developers.google.com/protocol-buffers/docs/overview).

#### `Features` Example

##### Syntax

As an example, take the following excerpt from
tensorflow/core/example/feature.proto.

<a name="feature-protobuf-code"></a>
```protobuf
// Containers to hold repeated fundamental values.
message BytesList {
  repeated bytes value = 1;
}
message FloatList {
  repeated float value = 1 [packed = true];
}
message Int64List {
  repeated int64 value = 1 [packed = true];
}

// Containers for non-sequential data.
message Feature {
  // Each feature can be exactly one kind.
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
};

message Features {
  // Map from feature name to feature.
  map<string, Feature> feature = 1;
};
```

Note that the `syntax = "proto3";` line in the `feature.proto` file indicates
that the proto3 (as opposed to proto2) version of the protobuf standard is
used.

`BytesList`, `FloatList`, `Int64List`, `Feature`, and `Features` are all
message types.

Each field in each message type definition has a unique numbered tag. For
example, the field `FloatList float_list` in message type `Feature` has a tag
of 2.

The `repeated` keyword indicates that a message may have zero or more of this
field, e.g. `BytesList` is made up of a sequence of zero or more `bytes`.

The `[packed = true]` syntax is a holdover from proto2, since `repeated` values
are using packed encoding by default in proto3. The meaning of packed encoding
can be found in the [protobuf encoding
guide](https://developers.google.com/protocol-buffers/docs/encoding#packed).

When an encoded message is parsed, missing fields are given [default
values](https://developers.google.com/protocol-buffers/docs/proto3#default).
For example, for numeric types such as `int64` the default value is zero.

Message types can be fields within other message types, for example
in `Feature`, `float_list` is a field of type `FloatList`.

A `oneof` field can be thought of as a `union` type in C, where each field in a
`oneof` block shares memory, and at most one field can be set at the same time.
Setting a `oneof` field will automatically clear all other members of the
`oneof` block, so if we have a `Feature` and set its `bytes_list` field, that
value will be deleted if we then set its `float_list` field.

The `map` syntax is a shortcut for generating key, value pairs, where the key
type is any integral or string type and the value can be anything. The
`map<string, Feature> feature = 1;` line uses this syntax, where the key is a
`string` and the value is a `Feature`. The `map` syntax from our example line
is equivalent to the code block below.

```protobuf
message MapFieldEntry {
        string key = 1;
        Feature value = 2;
}

repeated MapFieldEntry feature = 1;
```

So in our example, the `map` usage is making use of the fact that message type
definitions can be nested.

Based on the `.proto` file, the [protocol buffer
compiler](https://developers.google.com/protocol-buffers/docs/proto3#generating)
for Python will generate a module with a static descriptor of each message
type. This generated module is then used with the
`GeneratedProtocolMessageType` _metaclass_ to create the necessary Python data
access class at runtime.

More information about using the Python code generated
by the protobuf compiler can be found in the [protobuf Python
tutorial](https://developers.google.com/protocol-buffers/docs/pythontutorial).

More detailed information about proto3 syntax can be found in the [proto3
language guide](https://developers.google.com/protocol-buffers/docs/proto3).


##### Varints

To understand protobuf encoding we must first quickly cover varints. Protobufs
use base 128 varints, where the value is little-endian (i.e. the least
significant byte comes first), each byte contains 7 bits of the value and the
eighth bit (most-significant bit) is set to 1 if there are more bytes to come,
and 0 if it is the last byte.

For example, the value 300 is `ac 02`, where we get 300 by dropping the high
bit from the first byte `ac` to get `2c`, and shifting the second (and last)
byte down by 1 since we dropped a bit in the first byte: `2c 01`. `2c 01` read
in little-endian is 0x12c, or 300.

##### Encoding

We will use the following code snippet from the `write_tfrecords` function from
`read_mnist_data.py` as an example to explain protobuf encoding.

<a name="python-example-code"></a>
```python
image_raw = dataset.images[example_index].tostring()
feature = {
        'height': int64_feature(dataset.images.shape[1]),
        'width': int64_feature(dataset.images.shape[2]),
        'label': int64_feature(int(dataset.labels[example_index])),
        'image_raw': bytes_feature(image_raw)
}
next_features = tf.train.Features(feature = feature)
example = tf.train.Example(features = next_features)
writer.write(example.SerializeToString())
```

The `Features` protobuf is as given [above](#feature-protobuf-code) in the
syntax section, and the `Example` protobuf from
`tensorflow/core/example/example.proto` is a wrapper of `Features`, as shown
below.

```protobuf
message Example {
  Features features = 1;
};
```

Take the binary blob below, which was produced by the Python [code snippet
above](#python-example-code), and represents one `Example`.

```
00000000  0a d8 06 0a a4 06 0a 09  69 6d 61 67 65 5f 72 61  |........image_ra|
00000010  77 12 96 06 0a 93 06 0a  90 06 00 00 00 00 00 00  |w...............|
00000020  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
*
000000b0  00 00 03 12 12 12 7e 88  af 1a a6 ff f7 7f 00 00  |......~.........|
000000c0  00 00 00 00 00 00 00 00  00 00 1e 24 5e 9a aa fd  |...........$^...|
000000d0  fd fd fd fd e1 ac fd f2  c3 40 00 00 00 00 00 00  |.........@......|
000000e0  00 00 00 00 00 31 ee fd  fd fd fd fd fd fd fd fb  |.....1..........|
000000f0  5d 52 52 38 27 00 00 00  00 00 00 00 00 00 00 00  |]RR8'...........|
00000100  00 12 db fd fd fd fd fd  c6 b6 f7 f1 00 00 00 00  |................|
00000110  00 00 00 00 00 00 00 00  00 00 00 00 00 00 50 9c  |..............P.|
00000120  6b fd fd cd 0b 00 2b 9a  00 00 00 00 00 00 00 00  |k.....+.........|
00000130  00 00 00 00 00 00 00 00  00 00 00 0e 01 9a fd 5a  |...............Z|
00000140  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000150  00 00 00 00 00 00 00 00  00 8b fd be 02 00 00 00  |................|
00000160  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000170  00 00 00 00 00 0b be fd  46 00 00 00 00 00 00 00  |........F.......|
00000180  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000190  00 00 23 f1 e1 a0 6c 01  00 00 00 00 00 00 00 00  |..#...l.........|
000001a0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 51  |...............Q|
000001b0  f0 fd fd 77 19 00 00 00  00 00 00 00 00 00 00 00  |...w............|
000001c0  00 00 00 00 00 00 00 00  00 00 00 00 2d ba fd fd  |............-...|
000001d0  96 1b 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
000001e0  00 00 00 00 00 00 00 00  00 10 5d fc fd bb 00 00  |..........].....|
000001f0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000200  00 00 00 00 00 00 00 f9  fd f9 40 00 00 00 00 00  |..........@.....|
00000210  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000220  2e 82 b7 fd fd cf 02 00  00 00 00 00 00 00 00 00  |................|
00000230  00 00 00 00 00 00 00 00  00 00 27 94 e5 fd fd fd  |..........'.....|
00000240  fa b6 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000250  00 00 00 00 18 72 dd fd  fd fd fd c9 4e 00 00 00  |.....r......N...|
00000260  00 00 00 00 00 00 00 00  00 00 00 00 00 00 17 42  |...............B|
00000270  d5 fd fd fd fd c6 51 02  00 00 00 00 00 00 00 00  |......Q.........|
00000280  00 00 00 00 00 00 00 00  12 ab db fd fd fd fd c3  |................|
00000290  50 09 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |P...............|
000002a0  00 00 37 ac e2 fd fd fd  fd f4 85 0b 00 00 00 00  |..7.............|
000002b0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 88 fd  |................|
000002c0  fd fd d4 87 84 10 00 00  00 00 00 00 00 00 00 00  |................|
000002d0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
*
00000320  00 00 00 00 00 00 00 00  00 00 0a 0e 0a 05 6c 61  |..............la|
00000330  62 65 6c 12 05 1a 03 0a  01 05 0a 0f 0a 06 68 65  |bel...........he|
00000340  69 67 68 74 12 05 1a 03  0a 01 1c 0a 0e 0a 05 77  |ight...........w|
00000350  69 64 74 68 12 05 1a 03  0a 01 1c                 |idth.......|
```

The first byte of the binary encoding of a protobuf message type indicates the
field number and wire type of the message in the format `(field_number << 3) |
wire_type`. Based on the [table of wire type
values](https://developers.google.com/protocol-buffers/docs/encoding#structure),
our first byte `0a` corresponds to field number 1 and a Length-delimited wire
type. All embedded messages are length-delimited.

The next two bytes `d8 06` indicate the length in bytes of the entire `Example`
message, which is 0x358. Note that this matches the hexdump length once we take
away the three bytes we've already parsed.

The next three bytes `0a a4 06` indicate our next message type, which is the first
length-delimited  `Feature` field of `Features`, and is of length 0x324.

Recall that the `Feature` field is a key-value pair that was generated from the
`map<string, Feature>` syntax. So, the next bytes `0a 09 69 6d 61 67 65 5f 72
61 77` are a string type of length 9, which reads `'image_raw'`.

The proceeding field is a `BytesList`, with a nested `repeated bytes` type,
which is of length 784 or 28x28. This is the number of pixels in an MNIST
image, as expected. Note that the field number of the `BytesList` is now 2, so
its leading bytes is `2 << 3 | wire_type` or `12` instead of `1 << 3 |
wire_type` or `a0` as was the case with the `'image_raw'` field.

__Exercise:__ Parse the remaining `map<string, Feature>` fields of this
`Example`, using the [protobuf code](#feature-protobuf-code) as reference.
