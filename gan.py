import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

import os
import numpy as np


####################### WEIGHTS INITIALIZATION #############################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


####################### GENERATOR ###############################################################

class Generator(nn.Module):
    
    def __init__(self, nz, ngf, nc, ngpu):
        
        super(Generator, self).__init__()
        
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):

        return self.main(input)


################################## DISCRIMINATOR ########################################################

class Discriminator(nn.Module):
    
    def __init__(self, nc, ndf, ngpu):
        
        super(Discriminator, self).__init__()
        
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        
        x =  self.main(input)

        return x


########################################## TRANSFORMER ENCODER ####################################################

class FeedForwardLayer(nn.Module):
    
    def __init__(self, d_input, d_hidden, d_output):
        
        super(FeedForwardLayer, self).__init__()
        
        self.fc1 = nn.Linear(d_input, d_hidden)
        
        self.fc2 = nn.Linear(d_hidden, d_output)
        
        
    def forward(self, x):
        
        x = F.gelu(self.fc1(x))
        
        x = self.fc2(x)
        
        return x
        

class TransformerBlock(nn.Module):
    
    def __init__(self, d_input, d_hidden, d_output, num_heads=4):
        
        super(TransformerBlock, self).__init__()
        
        self.multihead_attention = nn.MultiheadAttention(d_input, num_heads=num_heads)
        
        self.norm1 = nn.LayerNorm(d_input)
        
        self.feed_forward = FeedForwardLayer(d_input, d_hidden, d_output)
        
        self.norm2 = nn.LayerNorm(d_output)
        
    def forward(self, x):
        
        attn_output, _ = self.multihead_attention(x, x, x)
        
        x = self.norm1(x + attn_output)
        
        ff_output = self.feed_forward(x)
        
        return ff_output


class TransformerEncoder(nn.Module):

    def __init__(self, d_input, d_hidden, d_output, num_blocks):

        super(TransformerEncoder, self).__init__()

        #THE ENCODER BLOCKS IN THE ARCHITECTURE
        self.encoder_blocks = [TransformerBlock( int(d_input/2**i) , d_hidden, int( d_input/2**(i+1)) ) for i in range(num_blocks)]

        self.encoder_blocks.append( TransformerBlock( int( d_input/2**num_blocks ), d_hidden, d_output) )
        
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)


    def forward(self, x):
        
        for block in self.encoder_blocks:
            
            x = block(x)

        return x



class SequenceClassifier(nn.Module):
    
    def __init__(self, d_input, n_classes):
        
        super(SequenceClassifier, self).__init__()
        
        self.d_input = d_input
        
        self.n_classes = n_classes
        
        self.classifier = nn.Linear(d_input, n_classes)

    def forward(self, x):
        
        predictions = []  # Initialize an empty list to store prediction

        for vector in x:
            
            prediction = self.classifier(vector)  # Get prediction
            
            predictions.append(prediction)

        predictions_tensor = torch.stack(predictions)  # Convert list to a tensor
        
        return predictions_tensor
            

################################## TRANSFORMER GENERATOR  ########################################
        

class TransformerGenerator(nn.Module):
    
    def __init__(self, d_input, d_embedding, d_hidden, d_output, num_blocks, ngf, nc, ngpu):
        
        super(TransformerGenerator, self).__init__()
        

        self.generator = Generator(d_output, ngf, nc, ngpu)
        

        self.embedding_layer = nn.Linear(d_input, d_embedding)
        
        
        self.encoder = TransformerEncoder(d_embedding, d_hidden, d_output, num_blocks)


        self.classifier = SequenceClassifier(d_output, 1)
        
        
        self.reconstruction_layer = nn.Linear(d_embedding, d_input)
        
    
    def forward(self, x):

        x = self.embedding_layer(x)
        
       

        x = self.encoder(x)

        
        predictions = self.classifier(x)
        
            
        return self.generator(x), predictions(x)

        