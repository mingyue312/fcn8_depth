import numpy as np
from math import ceil
import tensorflow as tf
import scipy.io as sio

VGG_MEAN = [103.939, 116.779, 123.68]

class Network:
    # defines the computation graph with structures laid out here, and parameters in the initializer
    def __init__(self, params, wd=5e-5, modelWeightPaths=None):
        self._params = params
        self._wd = wd
        self.modelDict = {}

        if modelWeightPaths is not None:
            # load all existing model .mat files. note that if there are duplicate names of weights, later appearances will
            # overwrite earlier ones
            for path in modelWeightPaths:
                self.modelDict.update(sio.loadmat(path))

    def build(self, inputData, keepProb=1.0):
        inputShape = tf.shape(inputData)

        # the input image at 5000x5000 is too big, so we downsample it to 2500x2500
        self.downsampled = tf.image.resize_bilinear(inputData, inputShape[1:3]/2)
        self.conv1_1 = self._conv_layer(self.downsampled, params=self._params["VGG16/conv1_1"])
        self.conv1_2 = self._conv_layer(self.conv1_1, params=self._params["VGG16/conv1_2"])
        self.pool1 = self._max_pool(self.conv1_2, 'VGG16/pool1')

        self.conv2_1 = self._conv_layer(self.pool1, params=self._params["VGG16/conv2_1"])
        self.conv2_2 = self._conv_layer(self.conv2_1, params=self._params["VGG16/conv2_2"])
        self.pool2 = self._max_pool(self.conv2_2, 'VGG16/pool2')

        self.conv3_1 = self._conv_layer(self.pool2, params=self._params["VGG16/conv3_1"])
        self.conv3_2 = self._conv_layer(self.conv3_1, params=self._params["VGG16/conv3_2"])
        self.conv3_3 = self._conv_layer(self.conv3_2, params=self._params["VGG16/conv3_3"])
        self.pool3 = self._max_pool(self.conv3_3, 'VGG16/pool3')

        self.conv4_1 = self._conv_layer(self.pool3, params=self._params["VGG16/conv4_1"])
        self.conv4_2 = self._conv_layer(self.conv4_1, params=self._params["VGG16/conv4_2"])
        self.conv4_3 = self._conv_layer(self.conv4_2, params=self._params["VGG16/conv4_3"])
        self.pool4 = self._max_pool(self.conv4_3, 'VGG16/pool4')

        self.conv5_1 = self._conv_layer(self.pool4, params=self._params["VGG16/conv5_1"])
        self.conv5_2 = self._conv_layer(self.conv5_1, params=self._params["VGG16/conv5_2"])
        self.conv5_3 = self._conv_layer(self.conv5_2, params=self._params["VGG16/conv5_3"])

        self.pool5 = self._max_pool(self.conv5_3, 'VGG16/pool5')

        self.fcn6 = self._conv_layer_dropout(self.pool5, params=self._params["fcn6"], keepProb=keepProb)
        self.fcn7 = self._conv_layer_dropout(self.fcn6, params=self._params["fcn7"], keepProb=keepProb)
        self.fcn8_coarse = self._conv_layer(self.fcn7, params=self._params["fcn8_coarse"])

        self.upscore2 = self._upscore_layer(self.fcn8_coarse, params=self._params["upscore2"],
                                           shape=tf.shape(self.pool4))

        self.score_pool4 = self._conv_layer(self.pool4, params=self._params["score_pool4"])

        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        self.upscore4 = self._upscore_layer(self.fuse_pool4, params=self._params["upscore4"],
                                            shape=tf.shape(self.pool3))

        self.score_pool3 = self._conv_layer(self.pool3, params=self._params["score_pool3"])

        self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

        # this is the raw output of the network
        self.upscore32 = self._upscore_layer(self.fuse_pool3, params=self._params["upscore32"],
                                             shape=tf.shape(inputData))

        # this is the argmax over channels (highest likelihood prediction)
        self.upscore32argmax = tf.argmax(self.upscore32, dimension=3)

        outputShape = tf.shape(self.upscore32)

        # network's softmax normalized output
        self.upscore32softmax = tf.reshape(self.upscore32, (-1, outputShape[3]))
        self.upscore32softmax = tf.nn.softmax(self.upscore32softmax)
        self.upscore32softmax = tf.reshape(self.upscore32softmax, outputShape)

    # LAYER BUILDING

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _average_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _conv_layer(self, bottom, params):
        # convolution layer definition
        with tf.variable_scope(params["name"]) as scope:
            filt = self.get_conv_filter(params)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(params)

            if params["act"] == "relu":
                activation = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            elif params["act"] == "lin":
                activation = tf.nn.bias_add(conv, conv_biases)
            elif params["act"] == "tanh":
                activation = tf.nn.tanh(tf.nn.bias_add(conv, conv_biases))
            elif params["act"] == "none":
                activation = tf.nn.bias_add(conv, conv_biases)

        return activation

    def _conv_layer_dropout(self, bottom, params, keepProb):
        # convolution layer definition that allows dropout. I think it may not be possible to branch on the presence of a
        # tf placeholder (keepProb), otherwise I would have merged this with _conv_layer(). I may be wrong though...
        with tf.variable_scope(params["name"]) as scope:
            filt = self.get_conv_filter(params)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(params)

            if params["act"] == "relu":
                activation = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            elif params["act"] == "lin":
                activation = tf.nn.bias_add(conv, conv_biases)
            elif params["act"] == "tanh":
                activation = tf.nn.tanh(tf.nn.bias_add(conv, conv_biases))

            activation = tf.nn.dropout(activation, keepProb)

        return activation

    # WEIGHTS GENERATION

    def get_bias(self, params):
        # defines the bias variable for a layer
        if params["name"]+"/biases" in self.modelDict:
            init = tf.constant_initializer(value=self.modelDict[params["name"]+"/biases"], dtype=tf.float32)
            print "loaded " + params["name"] + "/biases"
        else:
            init = tf.constant_initializer(value=0.0)
            print "generated " + params["name"] + "/biases"
        var = tf.get_variable(name="biases", initializer=init, shape=params["shape"][3])

        return var

    def get_conv_filter(self, params):
        # defines the convolution kernel variable for a layer
        if params["name"]+"/weights" in self.modelDict:
            init = tf.constant_initializer(value=self.modelDict[params["name"]+"/weights"], dtype=tf.float32)
            var = tf.get_variable(name="weights", initializer=init, shape=params["shape"])
            print "loaded " + params["name"]+"/weights"
        else:
            if params["std"]:
                stddev = params["std"]
            else:
                fanIn = params["shape"][0]*params["shape"][1]*params["shape"][2]
                stddev = (2.0/fanIn)**0.5

            init = tf.truncated_normal(shape=params["shape"], stddev=stddev)
            var = tf.get_variable(name="weights", initializer=init)

            print "generated " + params["name"]+"/weights, stddev=" + str(stddev)

        if not tf.get_variable_scope().reuse and self._wd: # weight decay. see lossFunction.py for more detailed description of "losses"
            weightDecay = tf.mul(tf.nn.l2_loss(var), self._wd,
                                  name='weight_loss')
            tf.add_to_collection('losses', weightDecay)

        return var

    def _upscore_layer(self, bottom, shape, params):
        # defines the upscoring convolution layer
        strides = [1, params["stride"], params["stride"], 1]
        with tf.variable_scope(params["name"]):
            in_features = bottom.get_shape()[3].value

            new_shape = [shape[0], shape[1], shape[2], params["outputChannels"]]
            output_shape = tf.pack(new_shape)

            f_shape = [params["ksize"], params["ksize"], params["outputChannels"], in_features]

            weights = self.get_deconv_filter(f_shape, params)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')
        return deconv

    def get_deconv_filter(self, f_shape, params):
        # gets the deconvolution filter for upscoring layer. if weights are not found in the dictionary, we initialize using bilinear filter weights
        if params["name"]+"/up_filter" in self.modelDict:
            init = tf.constant_initializer(value=self.modelDict[params["name"]+"/up_filter"], dtype=tf.float32)
            print "loaded " + params["name"] + "/up_filter"
        else:
            width = f_shape[0]
            height = f_shape[0]
            f = ceil(width / 2.0)
            c = (2 * f - 1 - f % 2) / (2.0 * f)
            bilinear = np.zeros([f_shape[0], f_shape[1]])
            for x in range(width):
                for y in range(height):
                    value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                    bilinear[x, y] = value
            weights = np.zeros(f_shape)
            for i in range(f_shape[2]):
                weights[:, :, i, i] = bilinear

            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)
            print "generated " + params["name"] + "/up_filter"
        return tf.get_variable(name="up_filter", initializer=init, shape=f_shape)
