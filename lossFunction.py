import tensorflow as tf

def modelTotalLoss(pred, gt, outputChannels=1):
    # this function returns the loss of our model to optimize.
    # pred is the raw output of the CNN without softmax normalization, gt is integer encoded ground truth

    # we first reshape the batch x imageHeight x imageWidth x channel prediction into (batch x imageHeight x imageWidth) x channel
    pred = tf.reshape(pred, (-1, outputChannels))

    # we transform the integer encoded ground truth into one-hot vectors by first squeezing the batch x imageHeight x imageWidth x 1
    # input into just a vector of length batch x imageHeight x imageWidth, and then generate the one_hot vector
    gt = tf.one_hot(indices=tf.to_int32(tf.squeeze(tf.reshape(gt, (-1, 1)))), depth=outputChannels, dtype=tf.float32)

    # define a small constant to avoid log(0)
    epsilon = tf.constant(value=1e-25)

    # perform softmax normalization across output channels of prediction
    predSoftmax = tf.to_float(tf.nn.softmax(pred))

    # define cross entropy at each pixel, which is equal to sum_classes{gt_k * log(max(predSoftmax, epsilon))}. Then, sum across the classes
    crossEntropy = -tf.reduce_sum(gt*tf.log(tf.maximum(predSoftmax, epsilon)), reduction_indices=[1])

    # calculate the mean entropy across all pixels
    crossEntropyMean = tf.reduce_mean(crossEntropy, name="cross_entropy_mean")

    # add the crossEntropyMean to a node in the computation graph named "losses". Note that weight decay penalties are also added to the same node
    tf.add_to_collection('losses', crossEntropyMean)

    # collect all losses together for total loss
    totalLoss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return totalLoss

def intersectionAndUnion(pred, gt, outputChannels):
    # calculates the intersection and union of prediction
    # pred is an integer encoded input of size batch x imageHeight x imageWidth x 1, where the integer at each pixel is the index of the max value
    # over the raw output channels from the CNN
    # gt is as defined in the modelTotalLoss function

    pred = tf.one_hot(indices=tf.argmax(tf.reshape(pred, (-1, outputChannels)), 1), depth=outputChannels, dtype=tf.int32)
    gt = tf.one_hot(indices=tf.to_int32(tf.squeeze(tf.reshape(gt, (-1, 1)))), depth=outputChannels, dtype=tf.float32)

    # convert both vectors to bool
    pred = tf.cast(pred, dtype=tf.bool)
    gt = tf.cast(gt, dtype=tf.bool)

    # intersection and union are simply AND and OR, respectively, followed by float casting and summation
    intersection = tf.reduce_sum(tf.to_float(tf.logical_and(pred, gt)), reduction_indices=[0])
    union = tf.reduce_sum(tf.to_float(tf.logical_or(pred, gt)), reduction_indices=[0])

    return intersection, union
