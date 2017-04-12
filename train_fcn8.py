import fcn8
from ioUtils import *
import lossFunction
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.io as sio
import re
import time
import skimage.io
from random import shuffle


VGG_MEAN = [103.939, 116.779, 123.68]
ROOT_PATH = "/home/ubuntu/yueming" #"/media/ming/DATADRIVE1/Academic/2017_Thesis" # "~/Downloads/TorontoCity"

def initialize_model(outputChannels, wd=None, modelWeightPaths=None):
    # initializes a FCN8 model using the parameters listed. For each layer (eg VGG16/conv1_1), there is an appropriate dictionary with layer specs
    params = {
        "VGG16/conv1_1": {"name": "VGG16/conv1_1", "shape": [3, 3, 5, 64], "std": None, "act": "relu"},
        "VGG16/conv1_2": {"name": "VGG16/conv1_2", "shape": [3, 3, 64, 64], "std": None, "act": "relu"},
        "VGG16/conv2_1": {"name": "VGG16/conv2_1", "shape": [3, 3, 64, 128], "std": None, "act": "relu"},
        "VGG16/conv2_2": {"name": "VGG16/conv2_2", "shape": [3, 3, 128, 128], "std": None, "act": "relu"},
        "VGG16/conv3_1": {"name": "VGG16/conv3_1", "shape": [3, 3, 128, 256], "std": None, "act": "relu"},
        "VGG16/conv3_2": {"name": "VGG16/conv3_2", "shape": [3, 3, 256, 256], "std": None, "act": "relu"},
        "VGG16/conv3_3": {"name": "VGG16/conv3_3", "shape": [3, 3, 256, 256], "std": None, "act": "relu"},
        "VGG16/conv4_1": {"name": "VGG16/conv4_1", "shape": [3, 3, 256, 512], "std": None, "act": "relu"},
        "VGG16/conv4_2": {"name": "VGG16/conv4_2", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "VGG16/conv4_3": {"name": "VGG16/conv4_3", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "VGG16/conv5_1": {"name": "VGG16/conv5_1", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "VGG16/conv5_2": {"name": "VGG16/conv5_2", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "VGG16/conv5_3": {"name": "VGG16/conv5_3", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},

        "fcn5": {"name": "fcn5", "shape": [1,1,512,512], "std": None, "act": "relu"},
        "fcn6": {"name": "fcn6", "shape": [1,1,512,4096], "std": None, "act": "relu"},
        "fcn7": {"name": "fcn7", "shape": [1,1,4096,4096], "std": None, "act": "relu"},
        "fcn8_coarse": {"name": "fcn8_coarse", "shape": [1,1,4096,outputChannels], "std": None, "act": "relu"},

        "upscore2": {"name": "upscore2", "ksize": 4, "stride": 2, "outputChannels": outputChannels},
        "score_pool4": {"name": "score_pool4", "shape": [1, 1, 512, outputChannels], "std": 1e-4, "act": "none"},
        "upscore4": {"name": "upscore4", "ksize": 4, "stride": 2, "outputChannels": outputChannels},
        "score_pool3": {"name": "score_pool3", "shape": [1, 1, 256, outputChannels], "std": 1e-4, "act": "none"},
        "upscore32": {"name": "upscore32", "ksize": 32, "stride": 16, "outputChannels": outputChannels}
        }

    return fcn8.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)

def forward_model(model, outputChannels, imageNamesListPath, imageDir, depthDir, outputSavePath=None):
    with tf.Session() as sess:
        tfBatchImages = tf.placeholder("float")
        keepProb = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print "attempting to build model"
            model.build(tfBatchImages, keepProb=keepProb)
            print "built the model"

        init = tf.global_variables_initializer() #initialize_all_variables() for 0.11 and before #global_variables_initializer() #for 0.12 and after

        sess.run(init)

        if outputSavePath and (not os.path.exists(outputSavePath)):
            os.makedirs(outputSavePath)

        imageNamesList = read_ids(imageNamesListPath)

        for imageName in imageNamesList:
            print "Processing " + imageName
            composed1 = skimage.io.imread(os.path.join(imageDir, imageName + '.tif'), plugin='tifffile').astype(
                float)
            composed2 = skimage.io.imread(os.path.join(depthDir, imageName + '.tif'), plugin='tifffile').astype(
                float)

            composed = np.dstack((composed1, composed2))
            image = (image_scaling(composed)).astype(float)

            outputImage = sess.run( [model.upscore32argmax], feed_dict={tfBatchImages: [image],
                                                               keepProb: 1.0})
            outputImage = outputImage[0][0]
            print(outputImage)
            print(len(outputImage))
            print(len(outputImage[0]))
            # saving output as a png file with uint8 encoding by multiplying class index by 50 so it's more visible using image viewers
            if outputSavePath:
                outputImage = (outputImage*50).astype(np.uint8)
                skimage.io.imsave(os.path.join(outputSavePath, imageName+'_zoning.png'), outputImage)

def train_model(model, outputChannels, learningRate, trainListPath, trainImageDir, trainDepthDir, trainGTDir, valListPath,
                valImageDir, valDepthDir, valGTDir, modelSavePath=None, savePrefix=None, initialIteration=1):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.device('/gpu:0'):
            tfBatchImages = tf.placeholder("float")
            tfBatchGT = tf.placeholder("float")
            keepProb = tf.placeholder("float")

            valImageNamesList = read_ids(valListPath)
            trainImageNamesList = read_ids(trainListPath)

            with tf.name_scope("model_builder"):
                print "attempting to build model"
                model.build(tfBatchImages, keepProb=keepProb)
                print "built the model"
            sys.stdout.flush()

            # model loss to be optimized over
            loss = lossFunction.modelTotalLoss(pred=model.upscore32, gt=tfBatchGT, outputChannels=outputChannels)

            # calculate intersection and union of prediction with respect to ground truth
            intersection, union = lossFunction.intersectionAndUnion(pred=model.upscore32, gt=tfBatchGT,
                                                                    outputChannels=outputChannels)

            # define training operation
            train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=loss)

            init = tf.global_variables_initializer() #initialize_all_variables() for 0.11 and before #global_variables_initializer() #for 0.12 and after
            sess.run(init)

            iteration = initialIteration

            while iteration < 1000:
                # perform validation
                totalIntersection = np.zeros(outputChannels)
                totalUnion = np.zeros(outputChannels)
                valLosses = []

                for imageName in valImageNamesList:
                    composed1 = skimage.io.imread(os.path.join(valImageDir, imageName + '.tif'), plugin='tifffile').astype(float)
                    composed2 = skimage.io.imread(os.path.join(valDepthDir, imageName + '.tif'), plugin='tifffile').astype(float)
                    # compose image
                    composed = np.dstack((composed1, composed2))
                    image = (image_scaling(composed)).astype(float)
                    gt = skimage.io.imread(os.path.join(valGTDir, imageName + '_labels.png')).astype(np.int32)
                    gt = gt[:, :, 0]
                    GT_KEYS_to_index(gt)

                    lossBatch, intersectionBatch, unionBatch= sess.run([loss, intersection, union],
                                                             feed_dict={tfBatchImages: [image],
                                                                        tfBatchGT: [gt],
                                                                        keepProb: 1.0})
                    # accumulate statistics across all validation images
                    valLosses.append(lossBatch)
                    totalIntersection += intersectionBatch
                    totalUnion += unionBatch

                if np.isnan(np.mean(valLosses)):
                    print "LOSS RETURNED NaN"
                    sys.stdout.flush()
                    return 1

                totalUnion[totalUnion < 1] = 1 # avoid divide by 0 error

                IoU = totalIntersection / totalUnion

                outputMessage = "%s Validation, loss: %.3f, " \
                      % (time.strftime("%H:%M:%S"), float(np.nanmean(valLosses)))

                for c in range(outputChannels):
                    outputMessage = outputMessage + " IoU Ch %d: %.3f"%(c, IoU[c])

                print outputMessage

                if (iteration % 5 == 0) or checkSaveFlag(modelSavePath):
                    modelSaver(sess, modelSavePath, savePrefix, iteration)

                sys.stdout.flush()

                shuffle(trainImageNamesList)

                for imageName in trainImageNamesList:
                    composed1 = skimage.io.imread(os.path.join(trainImageDir, imageName + '.tif'), plugin='tifffile').astype(float)
                    composed2 = skimage.io.imread(os.path.join(trainDepthDir, imageName + '.tif'), plugin='tifffile').astype(float)
                    #compose image
                    composed = np.dstack((composed1, composed2))
                    image = (image_scaling(composed)).astype(float)
                    gt = skimage.io.imread(os.path.join(trainGTDir, imageName + '_labels.png')).astype(np.int32)
                    gt = gt[:, :, 0]
                    GT_KEYS_to_index(gt)

                    # since images are 5000x5000, we may not have enough memory to perform the backpropagation pass with a whole image.
                    # in this case, we execute one training optimization step for each 1000x5000 strip of image. You can modify this feeding
                    # mechanism to suit your needs. Consider implementing a better feed structure that can randomly flip, rotate, or otherwise
                    # augment the training data.

                    for row in range(5):
                        imageStrip = image[row * 1000:(row + 1) * 1000, :, :]
                        imageBatch = []

                        for col in range(5):
                            imageBatch.append(imageStrip[:, col * 1000:(col + 1) * 1000, :])

                        gtStrip = gt[row*1000:(row+1)*1000, :]
                        gtBatch = []

                        for col in range(5):
                            gtBatch.append(gtStrip[:, col*1000:(col+1)*1000])

                        sess.run(train_op, feed_dict={tfBatchImages: imageBatch,
                                                      tfBatchGT: gtBatch,
                                                      keepProb: 0.7})
                iteration += 1

def modelSaver(sess, modelSavePath, savePrefix, iteration):
    # save all weights and biases in model using a dictionary format as a .mat file
    allWeights = {}

    for name in [n.name for n in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]:
        param = sess.run(name) # this will hold the raw names of all trainable variables (eg weights and biases)
        nameParts = re.split('[:/]', name)
        saveName = ""

        for part in [part for part in nameParts[0:-1]]: # this is to reformat the name into a clean and consistent format
            saveName += (part + "/")

        saveName = saveName[0:-1]

        allWeights[saveName] = param

    savePath = os.path.join(modelSavePath, savePrefix+'_%03d'%iteration)
    sio.savemat(savePath, allWeights)

    print "saving model to %s" % savePath

def checkSaveFlag(modelSavePath):
    # a way to trigger a save immediately upon next iteration externally. to activate, place a saveme.flag file at the modelSavePath
    flagPath = os.path.join(modelSavePath, 'saveme.flag')

    if os.path.exists(flagPath):
        return True
    else:
        return False

if __name__ == "__main__":
    outputChannels = 3 # number of channels for output. 3 semantic classes in this case
    savePrefix = "torontocity_fcn8_zoning3_halfres" # prefix of the saved models
    train = True # selects whether we wish to perform training with weights or just inference

    if train:
        learningRate = 1e-7
        wd = 5e-5

        # modelWeightPaths is a list of .mat files holding dictionaries of weight matrix keys, eg VGG16/conv2_2/weights. Later appearances of duplicate keys take precedence
        modelWeightPaths = [ROOT_PATH + "/test/aerial_semantic_models/fcn8_zoning/torontocity_fcn8_zoning3_halfres_120.mat"]

        initialIteration = 121

        model = initialize_model(outputChannels=outputChannels, wd=wd, modelWeightPaths=modelWeightPaths)

        train_model(model=model, outputChannels=outputChannels,
                    learningRate=learningRate,
                    trainListPath=ROOT_PATH + "/trainList.txt",
                    trainImageDir=ROOT_PATH + "/Images_tif",
                    trainDepthDir=ROOT_PATH + "/Depth",
                    trainGTDir=ROOT_PATH + "/Labels",
                    valListPath=ROOT_PATH + "/valList.txt",
                    valImageDir=ROOT_PATH + "/Images_tif",
                    valDepthDir=ROOT_PATH + "/Depth",
                    valGTDir=ROOT_PATH + "/Labels",
                    modelSavePath=ROOT_PATH + "/test/aerial_semantic_models/fcn8_zoning",
                    savePrefix=savePrefix,
                    initialIteration=initialIteration)

    else:
        modelWeightPaths = [ROOT_PATH + "/test/aerial_semantic_models/fcn8_zoning/torontocity_fcn8_zoning3_halfres_080.mat"]
        model = initialize_model(outputChannels=outputChannels, modelWeightPaths=modelWeightPaths)
        forward_model(model=model, outputChannels=outputChannels, imageNamesListPath=ROOT_PATH + "/valList.txt",
                      imageDir=ROOT_PATH + "/Images_tif",
                      depthDir=ROOT_PATH + "/Depth",
                      outputSavePath=ROOT_PATH + "/test/aerial_zoning_output/fcn8_zoning_output")
