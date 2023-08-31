import numpy as np
import os
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


############### LINEAR REGRESSION ############################

class LinearRegressor(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super(LinearRegressor, self).__init__()
        
        self.input_dim = input_dim
        
        self.output_dim = output_dim
        
        self.fcl = nn.Linear(input_dim, output_dim)
        
    
    def forward(self, x):
        
        return self.fcl(x)
    


################### FCNN (MULTY LAYER PERCEPTRON) ###################################"
    
class FCNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, depth, verbose=0):
        
        super(FCNN, self).__init__()
        
        self.input_dim = input_dim
        
        self.output_dim = output_dim
        
        self.depth = depth
        
        self.fcls = []
        
        for i in range(depth):
            
            #Check if we can add new layer (the dimensionality is devied by two)
            if input_dim // (2**(i+1)) > output_dim:
            
                self.fcls.append(nn.Linear(input_dim//(2**i), input_dim//(2**(i+1))))
                
            #if the dimensionality is less than two times the output we stop        
            else:
                
                self.depth = i
                
                if verbose == 1:
                
                    print("Depth changed do two output dimensionality, depth=", self.depth)
                
                break
                
        self.fcls.append(nn.Linear(input_dim//(2**(depth)), self.output_dim))
                
                
        self.fcls = nn.ModuleList(self.fcls)
        
    
    def forward(self, x):
        
        for fcl in self.fcls:
        
             x = fcl(x)
            
        return x



#################### CONVOLUTINAL NEURAL NETWORK ########################
    
    
class CNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, expected_latent_dim=None):
        
        super(CNN, self).__init__()
        
        self.input_dim = input_dim
        
        self.output_dim = output_dim
        
        if  expected_latent_dim:
            
            self.expected_latent_dim = expected_latent_dim
            
        else:
            
            self.expected_latent_dim = 32*(input_dim-8)
        
        self.conv1 = nn.Conv1d(1, 32, 5)
        
        self.conv2 = nn.Conv1d(32, 64, 5)
        
        self.pool = nn.MaxPool1d(2)
        
        self.fcl = nn.Linear(self.expected_latent_dim, output_dim)
        
    
    def forward(self, x):
        
        x = x.unsqueeze(dim=1)
        
        x = self.conv1(x)
        
        x = F.relu(self.conv2(x))
        
        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1)
        
        return self.fcl(x)


#################### ENCODER FOR COMBINED LOSS #################################
    

class Encoder(nn.Module):
    
    def __init__(self, input_dim, output_dim, depth):
        
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        
        self.output_dim = output_dim
        
        self.depth = depth
        
        self.fcls = []
        
        for i in range(depth):
            
            #Check if we can add new layer (the dimensionality is devied by two)
            if input_dim // (2**(i+1)) > output_dim:
            
                self.fcls.append(nn.Linear(input_dim//(2**i), input_dim//(2**(i+1))))
                
            #if the dimensionality is less than two times the output we stop        
            else:
                
                self.depth = i
                
                print("Depth changed do two output dimensionality, depth=", self.depth)
                
                break
                
        self.fcls.append(nn.Linear(input_dim//(2**(depth)), self.output_dim))
                
                
        self.fcls = nn.ModuleList(self.fcls)
        
    
    def forward(self, x):
        
        for fcl in self.fcls:
        
             x = fcl(x)
            
        return x


##################### REGRESSOR FOR THE MSE LOSS ##############################
    
class Regressor(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super(Regressor, self).__init__()
        
        self.input_dim = input_dim
        
        self.output_dim = output_dim
        
        self.fcl = nn.Linear(input_dim, output_dim)
        
        
    def forward(self, x):
        
        return self.fcl(x)
    
#################### CLASSIFIER TO KNOW IF THE MODEL CAN PREDICT IF THERE IS A MOUVEMENT #############################
    
class Classifier(nn.Module):
    
    def __init__(self, input_dim, n_classes):
        
        super(Classifier, self).__init__()
        
        self.input_dim = input_dim
        
        self.n_classes = n_classes
        
        self.fcl = nn.Linear(input_dim, n_classes)
        
        
    def forward(self, x):
        
        return F.sigmoid(self.fcl(x))
    

#################### MODEL FOR COMBINED LOSS ###########################

class RegClassModel(nn.Module):
    
    def __init__(self, input_dim, latent_dim, encoder_depth=2):
        
        super(RegClassModel, self).__init__()
        
        self.encoder = Encoder(input_dim, latent_dim, encoder_depth)
        
        self.regressor = Regressor(latent_dim, 1)
        
        self.classifier = Classifier(latent_dim, 1)
        
        
    def forward(self, x):
        
        x = self.encoder(x)
        
        reg = self.regressor(x)
        
        classifier = self.classifier(x)
        
        return reg, classifier


################## CNN ENCODER FOR COMBINED LOSS #######################


class ConvEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, expected_latent_dim=None):

        super(ConvEncoder, self).__init__()
        
        self.input_dim = input_dim
        
        self.output_dim = output_dim
        
        if  expected_latent_dim:
            
            self.expected_latent_dim = expected_latent_dim
            
        else:
            
            self.expected_latent_dim = 32*(input_dim-8)
        
        self.conv1 = nn.Conv1d(1, 32, 5)
        
        self.conv2 = nn.Conv1d(32, 64, 5)
        
        self.pool = nn.MaxPool1d(2)
        
    
    def forward(self, x):
        
        x = x.unsqueeze(dim=1)
        
        x = self.conv1(x)
        
        x = F.relu(self.conv2(x))
        
        x = self.pool(x)
        
        return torch.flatten(x, start_dim=1)


#################### COMBINED LOSS BUT WITH CNN ENCODER #########################


class ConvRegClassModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, expected_latent_dim=None):
        
        super(ConvRegClassModel, self).__init__()
        
        self.encoder = ConvEncoder(input_dim, output_dim, expected_latent_dim)
        
        self.regressor = Regressor(self.encoder.expected_latent_dim, 1)
        
        self.classifier = Classifier(self.encoder.expected_latent_dim, 1)
        
        
    def forward(self, x):
        
        x = self.encoder(x)
        
        reg = self.regressor(x)
        
        classifier = self.classifier(x)
        
        return reg, classifier