import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
                
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # dropout layer
        # self.dropout = nn.Dropout(0.4)
        
        # fully connected layer
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        
        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features, captions):
        # remove end-token from all captions
        embeddings = self.embed(captions[:,:-1])
    
        # concatenate captions embedidings and images features in one dimension array
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        # Pass the embeddings through the LSTM layer 
        out, hiddens = self.lstm(embeddings)
        
        # pass the output from LSTM layer through fully connected linear layer
        outputs = self.fc1(out)
    
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "         
        output = []
        
        for i in range(max_len):
            # pass the inputs to the LSTM layer
            out, hiddens = self.lstm(inputs, states)
            
            # pass the output from LSTM layer through fully connected linear layer
            out = self.fc1(out)
    
            # out.shape = [32, 1, 9955]
            # find the max value in the predicted vocabulary from the output tensor
            _, idx = out.max(2)
            print('Idx: ', idx.item())
            
            # update inputs and states
            inputs = self.embed(idx)
            states = hiddens

            # add the index to the output list
            output.append(idx.item())
                            
        return output