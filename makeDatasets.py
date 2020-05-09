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

if __name__=='__main__':

    for tt in range(5,6):
        print('The SEED is: ', tt)

    ################################################################################################
        dataFile   = "wcciExpData_ex99.csv"
        labelFile = "wcciExpLabel_ex99.csv"

        NoSamples = 250
        NoPatterns = 5
        NoisePct = 60
        torch.manual_seed( tt )

        # ORIGINAL TWO EXPERIMENTS
        # ex1 : seed,samp,patt,noise = 1500, 5, 10%, 117/tt
        # ex2 : seed,samp,patt,noise = 1500, 5, 25%, 117/tt

        # 3x3 PLOT OF EXPERIMENTS
        # ex91 : seed,samp,patt,noise = 1500, 10, 10%, 1    - lots of rules
        # ex92 : seed,samp,patt,noise = 1500, 10, 40%, 1
        # ex93 : seed,samp,patt,noise = 1500, 10, 10%, 9    - semantic confusion

        # ex94 : seed,samp,patt,noise = 1500,  3, 10%, 8    - few rules
        # ex95 : seed,samp,patt,noise = 1500,  3, 30%, 8
        # ex96 : seed,samp,patt,noise = 1500,  3,  5%, 7    - semantic confusion

        # ex97 : seed,samp,patt,noise =  250,  5, 10%, 5    - few data points
        # ex98 : seed,samp,patt,noise =  250,  5, 30%, 5
        # ex99 : seed,samp,patt,noise =  250,  5, 60%, 5    - high noise


    ################################################################################################

        # Data ---------------------------------------------------------------------
        x = torch.rand(NoSamples,2)
        print(x.shape[0])
        dl = torch.zeros(NoSamples)

        # Labels -------------------------------------------------------------------
        l = torch.zeros(NoSamples)
        # randomly pick the model parameters
        Membs = torch.rand(NoPatterns,2)        # the means
        Stds = torch.rand(NoPatterns,2)*(torch.rand(1)*0.6)      # the standard deviations
        Rows = torch.rand(NoPatterns,3)         # these are the linear rules weights
        # sample some random data
        dc = 0
        for r in range(NoPatterns):
            tm = torch.Tensor([Membs[r,0], Membs[r,1]])
            tc = torch.eye(2)
            tc[0,0] = Stds[r,0]*Stds[r,0]
            tc[1,1] = Stds[r,1]*Stds[r,1]
            tc[0,0] = tc[1,1]
            m = torch.distributions.multivariate_normal.MultivariateNormal(tm,tc)
            for i in range(int(NoSamples/NoPatterns)):
                x[dc,:] = m.sample()
                dl[dc] = r
                dc = dc + 1
        # now, have to fire each rule
        m = torch.zeros(NoSamples)
        ll = torch.rand(NoSamples)
        for i in range(NoSamples):
            mm = 0
            for r in range(NoPatterns):
                if( r == dl[i] ):
                   m[r] = 1
                mm = mm + m[r]
                ll[r] = x[i,0] * Rows[r,0] + x[i,1] * Rows[r,1] + Rows[r,2]
            for r in range(NoPatterns):
                l[i] = l[i] + ll[r]*m[r]
            l[i] = l[i] / (mm + 0.0000000000001)

        # go back in and make some of those points noisy
        #upper = 7
        #lower = 3
        BlakeRadius = 10
        NoNoise = int((NoSamples*NoisePct)/100)
        print(NoNoise)

        if(1):
            for r in range(NoPatterns):
                dc = 0 + (r*int(NoSamples/NoPatterns))
                tm = torch.Tensor([Membs[r,0], Membs[r,1]])
                tc = torch.eye(2)
                tc[0,0] = Stds[r,0]*Stds[r,0]*BlakeRadius
                tc[1,1] = Stds[r,1]*Stds[r,1]*BlakeRadius
                m = torch.distributions.multivariate_normal.MultivariateNormal(tm,tc)
                for i in range(int(NoNoise/NoPatterns)):
                    while(1):
                        asample = m.sample()
                        devamnt1 = abs( asample[0] - tm[0]  ) / (Stds[r,0])
                        devamnt2 = abs( asample[1] - tm[1]  ) / (Stds[r,1])
                        if( devamnt1 > 3 or devamnt2 > 3 ):
                            x[dc,0] = asample[0]
                            x[dc,1] = asample[1]
                            dc = dc + 1
                            break
                    #sign = 1
                    #if torch.rand(1) < 0.5:
                    #    sign = -1
                    #rando0 = torch.rand(1)*(upper-lower) + lower
                    #rando1 = torch.rand(1)*(upper-lower) + lower
                    #x[dc,0] = tm[0] + sign * rando0 * tc[0,0]
                    #x[dc,1] = tm[1] + sign * rando1 * tc[1,1]
                    #dc = dc + 1
            print('Done with building synthetic data set')

        # Output Data and Labels to CSV --------------------------------------------
        df_x = pd.DataFrame(np.asarray(x))
        df_l = pd.DataFrame(np.asarray(l))
        df_x.to_csv(dataFile, index=False, header=False)
        df_l.to_csv(labelFile, index=False, header=False)

    #######################################################

        # Visualizaiton ------------------------------------------------------------
        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.gca()
        data = np.asarray(x)
        NoSamples = len(x)
        print(len(x))

        colormap = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:yellow', \
                    'xkcd:cyan', 'xkcd:magenta', 'xkcd:olive', 'xkcd:orange',\
                    'xkcd:gold', 'xkcd:purple' ]
        shapemap = [ '.', 'p', '*', 's', 'o', '+']

        i = 0
        for r in range(NoPatterns):
            for d in range( int(NoNoise/NoPatterns) ):
                plt.plot( data[(int(NoSamples/NoPatterns))*r+d,0] , \
                          data[(int(NoSamples/NoPatterns))*r+d,1] , '.', \
                          color='xkcd:black', markeredgecolor='k', ms=5 )
            for d in range( int(NoNoise/NoPatterns), int(NoSamples/NoPatterns) ):
                plt.plot( data[(int(NoSamples/NoPatterns))*r+d,0] , \
                          data[(int(NoSamples/NoPatterns))*r+d,1] , '.', \
                          color='xkcd:cyan', markeredgecolor='k', ms=10 )
                i = i+1


        plt.show()
