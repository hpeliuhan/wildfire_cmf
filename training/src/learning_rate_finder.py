from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from FireDetectionNet import *
import tensorflow as tf
import click
import yaml
import os
from cmflib import cmf
from utils import *

class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        # store the model, stop factor, and beta value (for computing
        # a smoothed, average loss)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta

        # initialize our list of learning rates and losses,
        # respectively
        self.lrs = []
        self.losses = []

        # initialize our learning rate multiplier, average loss, best
        # loss found thus far, current batch number, and weights file
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

        #create metawriter for logging
        #self.metawriter=meta_writer

    def reset(self):
        # re-initialize all variables from our constructor
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self, data):
        # define the set of class types we will check for
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
             "DataFrameIterator", "Iterator", "Sequence"]

        # return whether our data is an iterator
        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.learning_rate)
        self.lrs.append(lr)

        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        # compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss

        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # stop returning and return from the method
            self.model.stop_training = True
            return

        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.learning_rate, lr)

    def find(self, trainData, startLR, endLR, epochs=None,
        stepsPerEpoch=None, batchSize=32, sampleSize=2048,
        classWeight=None, verbose=1):
        # reset our class-specific variables
        self.reset()
        #print("start learning rate: ", startLR)
        try:
            startLR = float(startLR)
            endLR = float(endLR)
        except ValueError:
            raise ValueError("startLR and endLR must be numeric values.")

        # determine if we are using a data generator or not
        useGen = self.is_data_iter(trainData)

        # if we're using a generator and the steps per epoch is not
        # supplied, raise an error
        if useGen and stepsPerEpoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)

        # if we're not using a generator then our entire dataset must
        # already be in memory
        elif not useGen:
            # grab the number of samples in the training data and
            # then derive the number of steps per epoch
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples / float(batchSize))

        # if no number of training epochs are supplied, compute the
        # training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        numBatchUpdates = epochs * stepsPerEpoch

        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

        # create a temporary file path for the model weights and
        # then save the weights (so we can reset the weights when we
        # are done)
        self.weightsFile = tempfile.mkstemp()[1]+".weights.h5"
        self.model.save_weights(self.weightsFile)

        # grab the *original* learning rate (so we can reset it
        # later), and then set the *starting* learning rate
        origLR = K.get_value(self.model.optimizer.learning_rate)
        #print(self.model.optimizer.learning_rate)
        K.set_value(self.model.optimizer.learning_rate, startLR)

        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs:
            self.on_batch_end(batch, logs))

        # check to see if we are using a data iterator
        if useGen:
            self.model.fit(
                trainData,
                steps_per_epoch=stepsPerEpoch,
                epochs=epochs,
                class_weight=classWeight,
                verbose=verbose,
                callbacks=[callback])

        # otherwise, our entire training data is already in memory
        else:
            # train our model using Keras' fit method
            self.model.fit(
                trainData[0], trainData[1],
                batch_size=batchSize,
                epochs=epochs,
                class_weight=classWeight,
                callbacks=[callback],
                verbose=verbose)

        # restore the original model weights and learning rate
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.learning_rate, origLR)

    def save_lr_loss(self, output_path):
        # Combine learning rates and losses into a single NumPy array
        data = np.array(list(zip(self.lrs, self.losses)))

        # Save the array to a .npy file
        np.save(output_path, data)

    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]

        # plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        
        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)


def learning_rate_finder(config_file:str, dir_config_file:str):

    params=yaml.safe_load(open(config_file))["learning_rate_finder"]
    rotation_range = float(params['aug']['rotation_range'])
    zoom_range = float(params['aug']['zoom_range'])
    width_shift_range = float(params['aug']['width_shift_range'])
    height_shift_range = float(params['aug']['height_shift_range'])
    shear_range = float(params['aug']['shear_range'])
    BATCH_SIZE=params["batches"]

    dir_config_file=yaml.safe_load(open(dir_config_file))["dir_config"]["learning_rate_finder"]
    input_dir1=dir_config_file["input1"]
    input_dir2=dir_config_file["input2"]
    outputdir=dir_config_file["output"]    
    os.makedirs(outputdir, exist_ok=True)

    trainX_path=os.path.join(input_dir1, "train_images.npy")
    trainY_path=os.path.join(input_dir1, "train_labels.npy")
    classWeight_path=os.path.join(input_dir1, "classWeight.npy")
    trainX= np.load(trainX_path)
    #trainX= tf.convert_to_tensor(trainX)
    trainY= np.load(trainY_path)
    #trainY= tf.convert_to_tensor(trainY)
    classWeight=np.load(classWeight_path)

    #load model
    model_path=os.path.join(input_dir2, "init_model.keras")
    model=tf.keras.models.load_model(model_path)
    aug=ImageDataGenerator(
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        horizontal_flip=params['aug']['horizontal_flip'],
        fill_mode=params['aug']['fill_mode'])

    lrf = LearningRateFinder(model)
    stepsPerEpochNum=np.ceil((trainX.shape[0] / float(BATCH_SIZE)))
    print("Steps per epoch:", stepsPerEpochNum, "Type:", type(stepsPerEpochNum))
    lr_epochs=params["epochs"]
    class_weight_dict = {i: weight for i, weight in enumerate(classWeight)}
    INIT_LR=params["start"]
    END_LR=params["end"]
    lrf.find(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        INIT_LR, END_LR,
        stepsPerEpoch=stepsPerEpochNum,
        epochs=lr_epochs,
        batchSize=BATCH_SIZE,
        classWeight=class_weight_dict)

    metawriter = set_cmf_environment("cmf","WILDFIRE")
    _ = metawriter.create_context(pipeline_stage="model training") 
    _ = metawriter.create_execution(execution_type="learning_rate_finder",custom_properties={"start_lr":INIT_LR,"epochs":lr_epochs}) 

    _ = metawriter.log_dataset(trainX_path,"input")
    _ = metawriter.log_dataset(trainY_path,"input")

    for ls in lrf.lrs:
        metawriter.log_metric("learning rate metrics",{"learning rate loss":float(ls)})
    _ = metawriter.commit_metrics("learning rate metrics")
    _ = metawriter.log_execution_metrics("learning rate summary",{"ave_loss":lrf.avgLoss,"best_loss":lrf.bestLoss})

    LRFIND_PLOT_PATH= os.path.join(outputdir,"lrfind_plot.png")
    lrf.plot_loss()
    plt.savefig(LRFIND_PLOT_PATH)
    print("Loss Rate finder complete")
    _ = metawriter.log_dataset(LRFIND_PLOT_PATH,'output')
    lr_loss_path=os.path.join(outputdir,"lr_loss.npy")
    lrf.save_lr_loss(lr_loss_path)
    _ = metawriter.log_dataset(lr_loss_path,'output')

@click.command()
@click.argument('config_file', required=True, type=str)
@click.argument('dir_config_file', required=True, type=str)
def learning_rate_finder_cli(config_file:str, dir_config_file: str) -> None:
    learning_rate_finder(config_file, dir_config_file)

if __name__=="__main__":
    learning_rate_finder_cli()

