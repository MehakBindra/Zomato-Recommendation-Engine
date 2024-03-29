#!/usr/bin/env python
# coding: utf-8

#Zomato Recommendation Engine

#Loading necessary packages and defining global variables
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from collections import defaultdict
import os
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd
from sys import platform

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")
OUTPUT_DIR_TRAIN='data/train.dat'
OUTPUT_DIR_TEST='data/test1.dat'

NUM_RESTS = 5138
NUM_USERS = 3579


# ## Restaurant reviews data
#
# We have reviews of 5138 restaurants from 3579 different users. Each user is given an ID from 0 to 3578 and similarly, each restaurant is given an ID from 0 to 5137. The format of data in the train.dat and test.dat file is:
#
#                         userID, restaurantID, rating
#
# This means each row contains a rating given to a particular restaurant by a given user.

# The function `get_sparse_mat` takes as the input the filename string which can either be `train.dat` and `test.dat` and constructs a sparse matrix containing the list of tuples for each user.
# The tuples in the list (restaurantID, rating) should have restaurantID as an integer value and rating as a float.



def get_sparse_mat(filename):

    '''

    Inputs:
        -filename: a string containing the name of the file from which we want
                    to extract the data. In our case it can be either train.dat
                    or test.dat

    Returns a python list of size 3579 (number of users) with each element of
    the list being a list of tuples (restaurantID, rating).

    '''

    sparse_mat = []

    f = open(filename)
    list=[]
    line=f.readline()
    m=line.split(',')
    x=m[0]
    for line in f:
        m=line.split(',')
        tup =(int(m[1]),float(m[2]))
        if(m[0]!=x):
            sparse_mat.append(list)
            x=m[0]
            list=[]
            list.append(tup)
        else:
            list.append(tup)

    sparse_mat.append(list)
    return sparse_mat


train_smat = get_sparse_mat(OUTPUT_DIR_TRAIN)
test_smat = get_sparse_mat(OUTPUT_DIR_TEST)

## Dataloaders
#
# Next we have defined a dataset class which can be used to efficiently iterate through the dataset.  Using the dataset objects we define data generators for train and test sets which are used to get batches of input data.

class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X_sam = torch.zeros(5138)
        y_sam = torch.zeros(5138)
        for i in range(len(self.X[index])):
            X_sam[self.X[index][i][0]] = self.X[index][i][1]

        for i in range(len(self.y[index])):
            y_sam[self.y[index][i][0]] = self.y[index][i][1]

        return X_sam, y_sam


train_dataset = Dataset(train_smat,train_smat)
test_dataset = Dataset(train_smat, test_smat)


params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6 if platform == 'linux' else 0}
training_generator = data.DataLoader(train_dataset, **params)
validation_generator = data.DataLoader(test_dataset, **params)


# ## Implementing Autoencoder Architecture
# The architeture of the network is:
#
# INPUT(size = 5138) -> FC+Tanh(size = 32) -> FC(size = 5138);

class DAE(nn.Module):
    def __init__(self):
        '''
        Define the layers and activation functions to be used in the network.
        '''
        super(DAE,self).__init__()

        # YOUR CODE HERE
        self.layer1 = nn.Linear(5138, 32)
        self.layer2=nn.Linear(32,64)
        self.layer3=nn.Linear(64,128)
        self.layer4 = nn.Linear(128, 5138)
        self.act=nn.ReLU()

    def forward(self, x):
        '''
        Implement the forward function which takes as input the tensor x and feeds it to the layers of the network
        and returns the output.

        Inputs:
            -x : Input tensor of shape [N_batch, 5138]

        Returns the output of neural network of shape [N_batch, 5138]
        '''

        out = torch.zeros(x.shape[0], 5138)

        y=self.layer1(x)
        y=y.tanh()
        y=self.layer2(y)
        y=self.act(y)
        y=self.layer3(y)
        y=self.act(y)
        out=self.layer4(y)

        return out

net = DAE()

# Now that we have defined our autoencoder network we need to define a loss function to train our model. We will be using mean squared error as our loss function which can be simply implemented by taking the squared sum of the errors between the model predictions and the labels and dividing it by the number of training examples. However, there is a small catch here.
#
# The function masked_loss takes as the input predictions and labels and calculates the mean squared error for the available ratings. One way of doing this is to first define a mask which is zero for the ratings not available and one for the available ones. Then we multiply this mask with the model predictions so that it zeros out the predictions of the network which are missing in the input data. Now we can calculate the sum of squared errors between the masked predictions and the input ratings and divide it with the number of available ratings which can be calculated by counting the number of ones in the mask.

def masked_loss(preds, labels):

    '''
    Inputs:
        -preds: Model predictions [N_batch, 5138]
        -labels: User ratings [N_batch, 5138]

    Returns the masked loss as described above.
    '''

    loss = 0

    N_batch=labels.shape[0]
    zeros=torch.zeros(N_batch,5138)
    ones=torch.ones(N_batch,5138)
    mask=torch.where(labels!=0,ones,zeros)

    mult=torch.mul(mask,preds)
    labels1=torch.mul(labels,-1)
    diff=torch.add(mult,labels1)
    diff=torch.pow(diff,2)
    count=0
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j]!=0:
                count=count+1
    loss=torch.div(torch.sum(diff),count)
    return loss


opti = optim.SGD(net.parameters(), lr=0.1)

#
# Now we have everything ready to start training our model. Each iteration will consist of 4 important steps:
#
# 1. Feeding input data to the network and obtaining model predictions
# 2. Compute the loss between inputs and labels say *loss*
# 3. Backpropagate the gradients using *loss.backward()*
# 4. Take the optimization step using the *step* method of the optimizer instance.

def train(net, criterion, opti, training_generator, validation_generator, max_epochs = 10):

    '''
    Inputs:
        - net: The model instance
        - criterion: Loss function, in our case it is masked_loss function.
        - opti: Optimizer Instance
        - training_generator: For iterating through the training set
        - validation_generator: For iterating through the test set
        - max_epochs: Number of training epochs. One epoch is defined as one complete presentation of the data set.

    Outputs:
        - train_losses: a list of size max_epochs containing the average loss for each epoch of training set.
        - val_losses: a list of size max_epochs containing the average loss for each epoch of test set.

        Note: We compute the average loss in an epoch by summing the loss at each iteration of that epoch
        and then dividing the sum by the number of iterations in that epoch.
    '''

    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        running_loss = 0 #Accumulate the loss in each iteration of the epoch in this variable
        cnt = 0 #Increment it each time to find the number iterations in the epoch.
        # Training iterations
        for batch_X, batch_y in training_generator:
            opti.zero_grad() #Clears the gradients of all variables.

            output=net(batch_X)
            loss=criterion(output,batch_y)
            loss.backward()
            opti.step()
            running_loss=running_loss+loss
            cnt=cnt+1

        print("Epoch {}: Training Loss {}".format(epoch+1, running_loss/cnt))
        train_losses.append(running_loss/cnt)


        #Now that we have trained the model for an epoch, we evaluate it on the test set
        running_loss = 0
        cnt = 0
        with torch.set_grad_enabled(False):
            for batch_X, batch_y in validation_generator:
                output=net(batch_X)
                loss=criterion(output,batch_y)
                running_loss=running_loss+loss
                cnt=cnt+1


        print("Epoch {}: Validation Loss {}".format(epoch+1, running_loss/cnt))

        val_losses.append(running_loss/cnt)

    return train_losses, val_losses

net = DAE()
opti = optim.SGD(net.parameters(), lr = 1e-1)
train_losses, val_losses = train(net, masked_loss, opti, training_generator, validation_generator, 20)


# Finally we plot the graphs for loss vs epochs.
plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epochs')


# Lets see how the network predictions compare with the actual ratings.

x, y = test_dataset.__getitem__(4)
pred = net(x)
print("Predicted Ratings: ", pred[y!=0].detach().numpy())
print("Actual Ratings: ", y[y!=0].numpy())


def get_predictions(net, train_data = train_smat):

    def get_test_smat(filename = 'data/test_hidden.dat'):
        sparse_dict = defaultdict(list)
        for line in open(filename):
            splitted_line = line.split(',')
            sparse_dict[int(splitted_line[0])].append((int(splitted_line[1])))

        sparse_mat = []
        sKeys = sorted(sparse_dict)
        for key in sKeys:
            sparse_mat.append(sparse_dict[key])

        return sparse_mat


    test_smat = get_test_smat()
    preds = []
    for i in range(len(train_data)):

        #Getting the actual vector from the sparse representation
        x = torch.zeros(5138)
        for j in range(len(train_data[i])):
            x[train_data[i][j][0]] = train_data[i][j][1]
        with torch.set_grad_enabled(False):
            pred = net(x).detach().numpy()

        pred = pred[test_smat[i]]
        user_rest_pred = np.concatenate([i*np.ones((len(pred),1),dtype=np.int),np.array(test_smat[i],dtype=np.int)[:,None], np.array(pred)[:,None]],axis = 1)
        preds += user_rest_pred.tolist()

    preds = np.array(preds)
    df = pd.DataFrame(preds)
    df[0] = df[0].astype('int')
    df[1] = df[1].astype('int')
    df[2] = df[2].astype('float16')
    df = df.drop(df.columns[[0, 1]], axis=1)
    df['index1'] = df.index.values
    df.columns = ['rating', 'id']
    df = df[['id','rating']]
    df.to_csv('predictions.csv', index=False, header=True)
    return df

df = get_predictions(net)
df.head()
