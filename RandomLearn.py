#!/usr/bin/env python
"""
Data Science and Machine Learning From First Principles: 
    Workshop 0 - Random Learning
    
"""
__author__ = "Jonathan Mann"

import numpy as np

class RandomLearn:
    """
    Test and update parameter weights based on random weight assignments

    Attributes:
        data :: [[int]] <- consists of input fields and a labels field
        trials :: int <- number of trials for generating random input weights
        ground_truth :: [float] <- actual values of weights that generated labels
        w :: [float] <- weights applied to parameters
        accuracy :: float <- accuracy of the weights for predicting the labels
        prior :: float <- probability of classification as true based on the label average
        prior_accuracy :: float <- accuracy prediction based on the prior
        param_len :: int <- number of parameter fields
        inputs :: [[int]] <- matrix of input parameters
        labels :: [[int]] <- labels vector
        iterations :: int <- number of cycles before arriving at final weights
    """
    def __init__(self,data_file="sample_data.csv",trials=50000,ground_truth=None):
        self.data = np.genfromtxt(data_file,delimiter=',')
        self.trials = trials
        self.ground_truth = ground_truth
        self.accuracy = 0
        self.process_data()
        self.train()

    def process_data(self):
        """split the data into inputs and labels"""
        
        # Count the input parameters by subtracting the labels column from the fields
        self.param_len = self.data.shape[1] - 1

        # Transpose the input fields and store
        self.inputs = self.data[:,:self.param_len].T

        # Transpose the labels field and store
        self.labels = self.data[:,self.param_len:].T

        # Calculate the prior
        self.prior = np.average(self.labels)

        # Use the prior to get a baseline accuracy
        self.prior_accuracy = np.around(max(self.prior,1 - self.prior),decimals=2)

        # Initialize weights to zero
        self.w = np.zeros((1,self.param_len))

    def train(self):
        """randomly test weights over the number of trials"""

        # Iterate over n random tests
        for i in range(self.trials):

            # Randomly assign test weights (limited to two decimal places)
            t_w = np.around(np.random.uniform(0,1,self.param_len),decimals=2)

            # Get the predictions made by the test weights
            preds = (np.dot(t_w,self.inputs) > .5)

            # Get accuracy of predictions
            t_acc = np.average((preds == self.labels))

            # Check the accuracy of the test weight predictions and update accuracy of model
            if t_acc > self.accuracy:
                self.accuracy = t_acc
                self.w = t_w
                if self.accuracy == 1:
                    break
        
        # Count the number of trials to get the final accuracy
        self.iterations = i + 1

    def output_findings(self):
        """output findings"""
        print("prior_accuracy:",self.prior_accuracy)
        if self.ground_truth is not None:
            print("ground_truth:",self.ground_truth)
        print("weights:",self.w)
        print("accuracy:",self.accuracy)
        print("iterations:",self.iterations)

if __name__=="__main__":
    rl = RandomLearn(data_file="sample_data.csv",trials=50000,ground_truth="[0.17,  0.22,  0,  0.36,  0.27]")
    rl.output_findings()
