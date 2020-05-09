##############################################################################
##############################################################################
##############################################################################
# University of Missouri-Columbia
#
# 7/5/2019
#
# Author: Blake Ruprecht and Muhammad Islam and Charlie Veal
#
# Description:
#  This is PyTorch code for an adaptive neural fuzzy inference system (ANFIS)
#  This file helps with managing data in support of mini batch
#
##############################################################################
#
# For more details, see:
# Jang, "ANFIS: adaptive-network-based fuzzy inference system," IEEE Transactions on Systems, Man and Cybernetics, 23 (3), 1993
#
##############################################################################
#
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
##############################################################################
##############################################################################
##############################################################################

import torch                                                                                # Library: Pytorch 
import torch.utils.data as utils                                                            # Library: Pytorch Dataset

class Format_Dataset(utils.Dataset):

    def __init__(self, data_params, choice):
        
        self.choice = choice 
        self.samples = torch.Tensor(data_params['samples'])                                 # Gather: Data Samples
        
        if(self.choice.lower() == 'train'): 
            self.labels = torch.Tensor(data_params['labels'])                               # Gather: Data Labels
        
    def __getitem__(self, index):                                                           
        
        if(self.choice.lower() == 'train'): 
            return self.samples[index], self.labels[index]                                  # Return: Next (Sample, Label) 
        else:
            return self.samples[index]                                                     

    def __len__(self):                                                                      
        
        return len(self.samples)
