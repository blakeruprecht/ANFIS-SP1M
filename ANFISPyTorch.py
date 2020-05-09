################################################################################
################################################################################
################################################################################
# Date:             MAY-05-2020
# Institution:      University of Missouri (Columbia, MO)
# Authors:          Blake Ruprecht, Muhammad Islam, and Derek Anderson
#
# DESCRIPTION ------------------------------------------------------------------
#    This is PyTorch code for an Adaptive Neural Fuzzy Inference System (ANFIS)
#
# Notes:
# 1. The below FuzzyNeuron class is a single fuzzy inference system (FIS)
#    What does that mean?
#    - Its a single first-order Takagi Sugeno Kang (TSK) inference system
#    What does that mean?
#    - Each neuron consists of R different IF-THEN rules and the aggregation of
#      their output.
# Coming soon...
#    - We will post cost updates that allow you to do things like
#    - Learn the number of rules R (via an algorithm like DBSCAN or
#      k-means/fcm/pcm with cluster validity)
#
# FOR MORE DETAILS, SEE: -------------------------------------------------------
#
#   Jang, "ANFIS: adaptive-network-based fuzzy inference system," IEEE
#       Transactions on Systems, Man and Cybernetics, 23 (3), 1993
#
# GNU GENERAL PUBLIC LICENSE ---------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


################################################################################
# LIBRARIES
################################################################################

# OUR LIBRARIES ----------------------------------------------------------------
from dataset_utils import Format_Dataset

# OTHER LIBRARIES --------------------------------------------------------------
import torch
from tqdm import tqdm
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from scipy.linalg import lstsq
import sys
import argparse
from torchvision import datasets
import pandas as pd


################################################################################
# ANFIS NEURON with TRAP M.F. and multiple different INITs
################################################################################

# ANFIS NEURON CLASS -----------------------------------------------------------
class FuzzyNeuron(torch.nn.Module):

    # Forward Pass -------------------------------------------------------------
    def forward(self,input):

        """ Forward Pass. This function evaluates an ANFIS rule base. If you
        don't already know, when working in PyTorch, keep all functions in
        PyTorch, so autograd will differentiate for you and you don't need to
        override backward pass with custom gradients. """

        batchSize = input.shape[0]
        z = torch.zeros( batchSize, self.R )                                    # our rule outputs
        ants = torch.zeros( self.R, batchSize, A )                              # our rule antecedents
        w = torch.ones( self.R, batchSize )                                     # our rule matching strengths

        sorted,indices = torch.sort(self.abcd)                                  # our sorted trapMF parameters

        for k in range( batchSize ):                                            # do for each sample, "k", in batchSize
            for r in range( self.R ):                                           # do for each rule, "r", in self.R
                z[k,r] = torch.dot(input[k,:],self.rho[r,:-1] ) \
                                                        + self.rho[r,self.A]    # the rule firing
                for n in range( self.A ):                                       # do for each antecedent, "n", in self.A
                    a,b,c,d = sorted[r,n,0],sorted[r,n,1],\
                              sorted[r,n,2],sorted[r,n,3]                       # the params for the trapMF come from the sort

                    ants[r,k,n] = torch.max( torch.tensor([ torch.min( \
                                 torch.tensor([((input[k,n]-a)/(b-a)), 1, \
                                 ((d-input[k,n])/(d-c))]) ), 0]) )             # trapezoidal Membership Function (trapMF)
                w[r,k] = torch.prod( ants[r,k,:] )


        # This is a proposed fix to the ants and w above, still in testing,
        # but we included it for usage. Let me know if you have any questions.
        #
        #             ants[r,k,n] = torch.clamp(torch.min((input[k,n]-a)/(b-a),\
        #                          (d-input[k,n])/(d-c) ), min=0.0, max=1.0)
        # w = torch.prod(ants,2)

        mul = torch.mm(z,w)                                                     # do their mult, but we only want the resultant diag
        diag = torch.diag(mul)                                                  # pull out the diag -> length equal to mini batch size
        wsum = torch.sum(w,dim=0)                                               # now sum across our weights (they are normalizers)
        out = diag / (wsum + 0.0000000000001)

        self.w = w
        self.z = z
        return out                                                              # now, do that normalization



    # Initialization -----------------------------------------------------------
    def __init__(self, R, A, InitMethod=0, TrainData=[], TrainLabels=[],chiReplace=-1):

        """ Init Function. This function will initialize the ANFIS parameters:
              (0) randomly
              (1) with k-means clustering
              (2) with k-means clustering and lstmsq rho guess """

        super(FuzzyNeuron,self).__init__()                                      # call parent function

        # InitMethod - Random --------------------------------------------------
        if( InitMethod == 0 ):

            self.R = R                                                          # number of rules
            self.A = A                                                          # number of antecedents
            self.abcd = torch.nn.Parameter(  torch.rand(R,A,4)  )               # trapMF parameters
            self.rho = torch.nn.Parameter( torch.rand(R,A+1) )                  # again, could be any +/-

            return

        # InitMethod - K-Means -------------------------------------------------
        elif( InitMethod == 1 ):

            self.R = R                                                          # number of rules
            self.A = A                                                          # number of antecedents

            NoSamples = TrainData.shape[0]                                      # run the K-Means clustering algorithm
            kmeans = KMeans(n_clusters=R, n_init=1, init='k-means++', \
                       tol=1e-6, max_iter=500, random_state=0).fit(TrainData)
            print('K-means mean guess')
            print(kmeans.cluster_centers_)

            mu = torch.rand(R,A)                                                # take the centers as our ant's
            for r in range(R):
                for n in range(A):
                    mu[r,n] = kmeans.cluster_centers_[r,n]

            sig = torch.rand(R,A)                                               # now, estimate the variances
            for r in range(R):
                inds = np.where( kmeans.labels_ == r )
                classdata = torch.squeeze( TrainData[inds,:] )
                for n in range(A):
                    sig[r,n] = torch.std( torch.squeeze(classdata[:,n]) )

            abcd = torch.zeros(R,A,4)                                           # build trap function from MU and SIGMA
            for r in range(R):
                for n in range(A):
                    abcd[r,n,0] = mu[r,n] - 5*sig[r,n]                          # c,b are +/- 2 stdDev from MU
                    abcd[r,n,1] = mu[r,n] - 2*sig[r,n]                          # d,a are +/- 5 stdDev from MU
                    abcd[r,n,2] = mu[r,n] + 2*sig[r,n]
                    abcd[r,n,3] = mu[r,n] + 5*sig[r,n]
            self.abcd = torch.nn.Parameter( abcd )

            self.rho = torch.nn.Parameter( torch.rand(R,A+1) )                  # random rhos, could be any +/-

            return

        # InitMethod - K-Means w/ Rho Guess ------------------------------------
        elif( InitMethod == 2 ):

            self.R = R                                                          # number of rules
            self.A = A                                                          # number of antecedents

            NoSamples = TrainData.shape[0]                                      # run the K-Means clustering algorithm
            kmeans = KMeans(n_clusters=R, n_init=1, init='k-means++', \
                       tol=1e-6, max_iter=500, random_state=0).fit(TrainData)
            #print('K-means mean guess')
            #print(kmeans.cluster_centers_)

            mu = torch.rand(R,A)                                                # take the centers as our ant's
            for r in range(R):
                for n in range(A):
                    mu[r,n] = kmeans.cluster_centers_[r,n]

            sig = torch.rand(R,A)                                               # now, estimate the variances
            for r in range(R):
                inds = np.where( kmeans.labels_ == r )
                classdata = torch.squeeze( TrainData[inds,:] )
                for n in range(A):
                    sig[r,n] = torch.std( torch.squeeze(classdata[:,n]) )

            abcd = torch.zeros(R,A,4)                                           # build trap function from MU and SIGMA
            for r in range(R):
                for n in range(A):
                    abcd[r,n,0] = mu[r,n] - 5*sig[r,n]                          # c,b are +/- 2 stdDev from MU
                    abcd[r,n,1] = mu[r,n] - 2*sig[r,n]                          # d,a are +/- 5 stdDev from MU
                    abcd[r,n,2] = mu[r,n] + 2*sig[r,n]
                    abcd[r,n,3] = mu[r,n] + 5*sig[r,n]

            Ws = torch.zeros(NoSamples,R,A)                                     # now, guess at rhos, using least means squared
            Wprods = torch.ones(NoSamples,R)                                    #  first, calc the constants using mu and sigma
            Wsums = torch.zeros(NoSamples) + 0.00000001
            for s in range(NoSamples):
                for i in range(R):
                    for j in range(A):
                        Ws[s,i,j] = max( min( min( (TrainData[s,j]-abcd[i,j,0])/(abcd[i,j,1]-abcd[i,j,0]), 1 ),\
                                             (abcd[i,j,3]-TrainData[s,j])/(abcd[i,j,3]-abcd[i,j,2]) ) , 0 )
                        Wprods[s,i] = Wprods[s,i] * Ws[s,i,j]                   # each rule is the t-norm (product here) of the MFs
                    Wsums[s] = Wsums[s] + Wprods[s,i]                           # we will normalize by this below

            CoeffMatrix = np.zeros((NoSamples,R*(A+1)))                         # make up our matrix to solve for
            for s in range(NoSamples):
                ctr = 0
                for i in range(R):
                    for j in range(A):
                        CoeffMatrix[s,ctr] = Wprods[s,i] * TrainData[s,j]\
                                             * ( 1.0 / Wsums[s] )
                        ctr = ctr + 1
                    CoeffMatrix[s,ctr] = Wprods[s,i] * 1.0 * (1.0 / Wsums[s])   # now, do for bias term
                    ctr = ctr + 1

            p, res, rnk, s = lstsq(CoeffMatrix, TrainLabels)                    # solve for rho
            rho = torch.zeros(R,A+1)                                            # format and return this as our init
            ctr = 0
            #print(p[ctr].dtype)

            for i in range(R):
                for j in range(A+1):
                    #rho[i,j] = torch.from_numpy(p[ctr])
                    rho[i,j] = p[ctr]
                    ctr = ctr + 1
            #print('Rho guess is')
            #print(rho)


            self.abcd = torch.nn.Parameter( abcd, requires_grad=True )
            self.rho = torch.nn.Parameter( rho, requires_grad=True )

            return


################################################################################
# TRAIN and TEST
################################################################################

if __name__=='__main__':

    # PARAMETERS -----------------------------------------------------------
    EXPERIMENT = 1      # 0 = ANFIS, 1 = Pre-Process
    NoEpochs = 5
    LEARNING_RATE = 1e-1


    # LOAD IN DATASETS AND TYPICALITIES ------------------------------------
    EXPERIMENT = 1     # 0 = ANFIS, 1 = PREPROCESS, 2 = GRADIENT SCALE
    num = 91
    chiReplace = 0      # 0 = NOTHING, 1 = ANTECEDENT, 2 = CONSEQUENT, 3 = AGGREGATION
    dataFile   = "wcciExpData_ex" + str(num) + ".csv"                       # load in data matrix
    labelFile = "wcciExpLabel_ex" + str(num) + ".csv"                       # load in label matrix
    typicFile = "wcciExpTypic_ex" + str(num) + ".csv"                       # load in typicality matrix

    df_x = pd.read_csv( dataFile, header=None)
    df_l = pd.read_csv( labelFile, header=None)
    df_t = pd.read_csv( typicFile, header=None)
    x = torch.tensor(df_x.values)
    x = x.float()
    l = torch.tensor(df_l.values)
    l = l.float()
    l = torch.squeeze(l)
    t = torch.tensor(df_t.values)
    t = t.t()

    NoAnts = 2
    NoPatterns = len(t[0,:])
    print("The number of rules identified: ", NoPatterns)
    NoRules = NoPatterns
    x_noise = []
    TYP_THRESHOLD = 0.2


    # PRE-PROCESS DATA -----------------------------------------------------
    if EXPERIMENT == 1:
        x_train = x.numpy()
        l_train = l.numpy()
        t_train = t.numpy()
        badIdx = []

        for i in range(len(x_train)):
            if torch.max(t[i,:]) < TYP_THRESHOLD:
                badIdx = np.append(badIdx, int(i))

        badIdx = badIdx.astype(int)
        x_noise = x_train[badIdx]
        x_train = np.delete(x_train, badIdx, axis=0)
        x_train = torch.from_numpy(x_train)
        l_noise = l_train[badIdx]
        l_train = np.delete(l_train, badIdx, axis=0)
        l_train = torch.from_numpy(l_train)
        t_noise = t_train[badIdx]
        t_train = np.delete(t_train, badIdx, axis=0)
        t_train = torch.from_numpy(t_train)

    if EXPERIMENT != 1:
        x_train = x
        l_train = l
        t_train = t
    dataset = {'samples': x_train, 'labels': l_train}
    train = Format_Dataset(dataset, choice='Train')
    train = torch.utils.data.DataLoader(shuffle=True, dataset=train,
                                            batch_size = 15)



    # TRAINING -------------------------------------------------------------

    net = FuzzyNeuron(NoRules,NoAnts,2,x_train,l_train)    # create the ANFIS neuron
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, \
                betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    totalLoss = 0

    for epoch in tqdm(range( NoEpochs ),'Training Epoch'):
        i=0
        for sample, label in train:
            outputs = net(sample)
            loss = criterion(outputs, label)
            totalLoss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i=i+1
        print("---Loss for epoch: ",epoch," is: ",totalLoss)
        totalLoss = 0




    # VISUALIZATION of Datapoints ------------------------------------------
    plt.figure(num=1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    data = np.asarray(x)
    NoSamples = len(x)
    print(len(x))
    print("Number of Data Points: ", len(x))
    print("Number of Test Points: ", len(x_train))
    print("Number of Noisy Points: ", len(x_noise))
    # ABCD = (net.abcd.data).cpu().numpy()

    colormap = ['xkcd:slate blue', 'xkcd:olive', 'xkcd:light purple', 'xkcd:cerulean', 'xkcd:tan',\
                'xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:light blue', 'xkcd:teal']
    shapemap = [ '.', 'p', '*', 's', 'o', '+']

    i = 0
    for r in range(NoPatterns):
        for d in range( int(NoSamples/NoPatterns) ):
            if torch.max(t[i,:]) < TYP_THRESHOLD:
                plt.plot( data[(int(NoSamples/NoPatterns))*r+d,0] , \
                          data[(int(NoSamples/NoPatterns))*r+d,1] , '.', \
                          color=colormap[r], markeredgecolor='k', \
                          ms = (15*torch.max(t[i,:])+2) )
                i = i+1
            else:
                plt.plot( data[(int(NoSamples/NoPatterns))*r+d,0] , \
                          data[(int(NoSamples/NoPatterns))*r+d,1] , '.', \
                          color=colormap[r], markeredgecolor='k', \
                          ms = (15*torch.max(t[i,:])+2) )
                i = i+1


    # VISUALIZATION of Rules -----------------------------------------------
    # ABCD = (net.abcd.data).cpu().numpy()
    # boxcolors = ['xkcd:magenta','xkcd:green']
    #
    # for r in range(NoRules):
    #     coreXY = (ABCD[r,0,1],ABCD[r,1,1])
    #     coreHt = ABCD[r,1,2] - ABCD[r,1,1]
    #     coreWd = ABCD[r,0,2] - ABCD[r,0,1]
    #     wideXY = (ABCD[r,0,0],ABCD[r,1,0])
    #     wideHt = ABCD[r,1,3] - ABCD[r,1,0]
    #     wideWd = ABCD[r,0,3] - ABCD[r,0,0]
    #     coreRect = matplotlib.patches.Rectangle(coreXY,coreWd,coreHt,\
    #                         color=boxcolors[ex], ls='-', lw=2, fill=False, zorder=489+ex)
    #     wideRect = matplotlib.patches.Rectangle(wideXY,wideWd,wideHt,\
    #                         color=boxcolors[ex], ls='--', lw=2, fill=False, zorder=489+ex)
    #     ax.add_patch(coreRect)
    #     ax.add_patch(wideRect)


    plt.show()
