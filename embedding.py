import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch.nn.utils as utils
import torch.nn.init as init


############# AUTOENCODER ############################################

class Encoder(nn.Module):

    def __init__(self, n_features, embedding_dim, n_layers):

        assert n_features > embedding_dim, "The dimension of the features should be greater than the embedding dimension"

        super(Encoder, self).__init__()

        self.n_features = n_features

        self.embedding_dim = embedding_dim

        self.n_layers = n_layers

        self.ratio = (n_features - embedding_dim)/n_layers

        self.fcls = nn.ModuleList([nn.Linear( n_features - int(i*self.ratio), n_features - int((i+1) * self.ratio)) for i in range(n_layers)])
        

    def forward(self, x):

        for fcl in self.fcls:

            x = fcl(x)

        return x


class Decoder(nn.Module):

    def __init__(self, n_features, embedding_dim, n_layers):

        assert n_features > embedding_dim, "The dimension of the features should be greater than the embedding dimension"

        super(Decoder, self).__init__()

        self.n_features = n_features

        self.embedding_dim = embedding_dim

        self.n_layers = n_layers

        self.ratio = (n_features - embedding_dim)/n_layers

        self.fcls = nn.ModuleList([nn.Linear( embedding_dim + int(i*self.ratio), embedding_dim + int((i+1) * self.ratio)) for i in range(n_layers)])
        

    def forward(self, x):

        for fcl in self.fcls:

            x = fcl(x)

        return x


class AutoEncoder(nn.Module):

    def __init__(self, n_features, embedding_dim, n_layers):
        
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(n_features, embedding_dim, n_layers)

        self.decoder = Decoder(n_features, embedding_dim, n_layers)


    def forward(self, x):

        x = self.encoder(x)

        return self.decoder(x)



######### RECURRENT AUTOENCODER #############################################""


class RecurrentEncoder(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim, n_layers):
        
        super(RecurrentEncoder, self).__init__()

        self.seq_len = seq_len
        
        self.n_features = n_features
        
        self.embedding_dim = embedding_dim
        
        self.hidden_dim = 2 * embedding_dim
        
        self.n_layers = n_layers

        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=self.n_layers,
          batch_first=True  
        )
        
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size= self.embedding_dim,
          num_layers=self.n_layers,
          batch_first=True
        )

    def forward(self, x):
        
        x, (_, _) = self.rnn1(x) 
        
        x, (hidden_n, _) = self.rnn2(x)
        
        return hidden_n.reshape((x.shape[0], self.embedding_dim))

class RecurrentDecoder(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim, n_layers):
        
        super(RecurrentDecoder, self).__init__()

        self.seq_len = seq_len

        self.embedding_dim = embedding_dim

        self.n_layers = n_layers
        
        self.hidden_dim = 2 * self.embedding_dim
        
        self.n_features = n_features

        self.rnn1 = nn.LSTM(
        input_size=self.embedding_dim,
        hidden_size=self.embedding_dim,
        num_layers=self.n_layers,
        batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
        input_size=self.embedding_dim,
        hidden_size=self.hidden_dim,
        num_layers=self.n_layers,
        batch_first=True
        )
        
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.unsqueeze(1)

        x = x.expand(-1, self.seq_len, -1)

        x = x.reshape((batch_size, self.seq_len, self.embedding_dim))
        
        x, (hidden_n, cell_n) = self.rnn1(x)
        
        x, (hidden_n, cell_n) = self.rnn2(x)
        
        return self.output_layer(x)
        

class RecurrentAutoencoder(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim, n_layers):
        
        super(RecurrentAutoencoder, self).__init__()
        
        self.seq_len = seq_len
        
        self.n_features = n_features

        self.embedding_dim = embedding_dim

        self.n_layers = n_layers

        self.encoder = RecurrentEncoder(seq_len, n_features, embedding_dim, n_layers)
        
        self.decoder = RecurrentDecoder(seq_len, n_features, embedding_dim, n_layers)

    
    def forward(self, x):
        
        x = self.encoder(x)

        x = self.decoder(x)
        
        return x


##################### MOUVEMENT CLASSIFIER #####################################

class Classifier(nn.Module):

    def __init__(self, in_dim, n_classes):

        super(Classifier, self).__init__()

        self.in_dim = in_dim

        self.n_classes = n_classes

        self.fcl = nn.Linear(in_dim, n_classes)

        self.batch_norm = nn.BatchNorm1d(in_dim)

        init.xavier_uniform_(self.fcl.weight)


    def forward(self, x):

        x = self.batch_norm(x)

        x = self.fcl(x)

        return x


######################## AUTOENCODER WITH MOUVEMENT INFERENCE ###########################

class Inference_AutoEncoder(nn.Module):

    def __init__(self, n_features, embedding_dim, n_layers, n_classes):
        
        super(Inference_AutoEncoder, self).__init__()

        self.encoder = Encoder(n_features, embedding_dim, n_layers)

        self.clf = Classifier(embedding_dim, n_classes)

        self.decoder = Decoder(n_features, embedding_dim, n_layers)


    def forward(self, x):

        x = self.encoder(x)

        prob = self.clf(x)

        return self.decoder(x), prob


######################## RECURRENT AUTOENCODER WITH MOUVEMENT INFERENCE ###########################


class Inference_RecurrentAutoencoder(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim, n_layers, n_classes):
        
        super(Inference_RecurrentAutoencoder, self).__init__()
        
        self.seq_len = seq_len
        
        self.n_features = n_features

        self.embedding_dim = embedding_dim

        self.n_layers = n_layers

        self.encoder = RecurrentEncoder(seq_len, n_features, embedding_dim, n_layers)

        self.clf = Classifier(embedding_dim, n_classes)
        
        self.decoder = RecurrentDecoder(seq_len, n_features, embedding_dim, n_layers)

    
    def forward(self, x):
        
        x = self.encoder(x)

        prob = self.clf(x)

        x = self.decoder(x)
        
        return x, prob




###################### ATTENTION AUTOENCODER ###################################

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


class TransformerDecoder(nn.Module):

    def __init__(self, d_input, d_hidden, d_output, num_blocks):

        super(TransformerDecoder, self).__init__()

        
        #THE DECODER BLOCKS 
        self.decoder_blocks = [TransformerBlock( d_output, d_hidden, int( d_input/2**num_blocks ), num_heads=1 )] 

        for i in range(num_blocks):
        
            self.decoder_blocks.append( TransformerBlock( int(d_input/2**(num_blocks-i)), d_hidden, int(d_input/2**(num_blocks-i-1)) ) )
        
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)



    def forward(self, x):

        for block in self.decoder_blocks:
            
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
            

        

class TransformerDR(nn.Module):
    
    def __init__(self, d_input, d_embedding, d_hidden, d_output, num_blocks):
        
        super(TransformerDR, self).__init__()

        self.embedding_layer = nn.Linear(d_input, d_embedding)

        
        self.encoder = TransformerEncoder(d_embedding, d_hidden, d_output, num_blocks)

        self.decoder = TransformerDecoder(d_embedding, d_hidden, d_output, num_blocks)

        self.classifier = SequenceClassifier(d_output, 1)
        
        
        self.reconstruction_layer = nn.Linear(d_embedding, d_input)
        
    
    def forward(self, x):

        x = self.embedding_layer(x)
        
       

        x = self.encoder(x)

        
        predictions = self.classifier(x)


        
        x = self.decoder(x)
        
            
        reconstructed_x = self.reconstruction_layer(x)
        
        
        return reconstructed_x, predictions