#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data


# In[2]:


#MODEL
class da_mil(nn.Module):
    def __init__(self,in_dim,T,encoder_hidden_size,decoder_hidden_size,fc_hidden_size,out_dim):
        super(da_mil, self).__init__()
        self.input_size=in_dim
        self.T=T
        self.encoder_hidden_size=encoder_hidden_size
        self.decoder_hidden_size=decoder_hidden_size
        self.encoder_lstm_layer = nn.LSTM(input_size = in_dim, hidden_size = self.encoder_hidden_size, num_layers = 1)
        self.encoder_attn_linear = nn.Linear(in_features = 2 * self.encoder_hidden_size + T, out_features = 1)
        self.decoder_lstm_layer=nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size)
#         self.decoder_attn_linear=nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
#                                                  nn.Tanh(),nn.Dropout(0.5),nn.Linear(encoder_hidden_size, 1))
        self.decoder_attn_linear=nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                                 nn.Tanh(),nn.Linear(encoder_hidden_size, 1))
        self.decoder_fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc = nn.Sequential(nn.Linear(decoder_hidden_size, fc_hidden_size), nn.Tanh()).cuda()

        self.logitic = nn.Sequential(nn.Linear(fc_hidden_size, out_dim), nn.Sigmoid()).cuda()

    def forward(self,related_seq,target_seq):
        input_final=Variable(torch.randn(BATCH_SIZE,step,decoder_hidden_size)).cuda()
        ####多步
        for i in range(step):
            X=related_seq[:,i:i+self.T,:]
            Y=target_seq[:,i:i+self.T]
            input_encoded = Variable(X.data.new(X.size(0), self.T, self.encoder_hidden_size).zero_())
            #encoder
            # hidden, cell: initial states with dimention hidden_size
            hidden = Variable(X.data.new(1, X.size(0), self.encoder_hidden_size).zero_()) # 1 * batch_size * hidden_size
            cell =Variable(X.data.new(1, X.size(0), self.encoder_hidden_size).zero_())
            for t in range(self.T):
                x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim = 2) # batch_size * input_size * (2*hidden_size + T - 1)
                
                x = self.encoder_attn_linear(x.view(-1, self.encoder_hidden_size * 2 + self.T)) # (batch_size * input_size) * 1
                attn_weights = F.softmax(x.view(-1, self.input_size)) # batch_size * input_size, attn weights with values sum up to 1.
                weighted_input = torch.mul(attn_weights, X[:, t, :]) # batch_size * input_size

                self.encoder_lstm_layer.flatten_parameters() 
                _,lstm_states = self.encoder_lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
                hidden = lstm_states[0]
                cell = lstm_states[1]
                # Save output
                input_encoded[:, t, :] = hidden
        
        
            #decoder
            hidden = Variable(input_encoded.data.new(1, X.size(0), self.decoder_hidden_size).zero_())
            cell = Variable(input_encoded.data.new(1, X.size(0), self.decoder_hidden_size).zero_())
            
            for t in range(self.T ):
                ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
                x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
                x = F.softmax(self.decoder_attn_linear(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T)) # batch_size * T - 1, row sum up to 1
                context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size
                if t < self.T:
                    y_tilde = self.decoder_fc(torch.cat((context, Y[:, t].unsqueeze(1)), dim = 1)) # batch_size * 1
                    self.decoder_lstm_layer.flatten_parameters()
                    _, lstm_output = self.decoder_lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                    hidden = lstm_output[0] # 1 * batch_size * decoder_hidden_size
                    cell = lstm_output[1] # 1 * batch_size * decoder_hidden_size
            input_final[:,i,:] = hidden[0]
        
        #mil
        out_fc = self.fc(input_final)
        out_p = self.logitic(out_fc)

        p_max = torch.max(out_p, 1)[0].cuda()
        return p_max
    
    def evaluate(self,related_seq,target_seq):
        input_final=Variable(torch.randn(BATCH_SIZE2,step,decoder_hidden_size)).cuda()
        ####多步
        for i in range(step):
            X=related_seq[:,i:i+self.T,:]
            Y=target_seq[:,i:i+self.T]
            input_encoded = Variable(X.data.new(X.size(0), self.T , self.encoder_hidden_size).zero_())

            #encoder
            # hidden, cell: initial states with dimention hidden_size
            hidden = Variable(X.data.new(1, X.size(0), self.encoder_hidden_size).zero_()) # 1 * batch_size * hidden_size
            cell =Variable(X.data.new(1, X.size(0), self.encoder_hidden_size).zero_())
            for t in range(self.T):
                # Eqn. 8: concatenate the hidden states with each predictor
                x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim = 2) # batch_size * input_size * (2*hidden_size + T - 1)

                x = self.encoder_attn_linear(x.view(-1, self.encoder_hidden_size * 2 + self.T)) # (batch_size * input_size) * 1
                attn_weights = F.softmax(x.view(-1, self.input_size)) # batch_size * input_size, attn weights with values sum up to 1.

                weighted_input = torch.mul(attn_weights, X[:, t, :]) # batch_size * input_size
                self.encoder_lstm_layer.flatten_parameters() 
                _,lstm_states = self.encoder_lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
                hidden = lstm_states[0]
                cell = lstm_states[1]
                # Save output
                input_encoded[:, t, :] = hidden
        
        
            #decoder
            hidden = Variable(input_encoded.data.new(1, X.size(0), self.decoder_hidden_size).zero_())
            cell = Variable(input_encoded.data.new(1, X.size(0), self.decoder_hidden_size).zero_())
            

            for t in range(self.T ):
                x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
                x = F.softmax(self.decoder_attn_linear(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T)) # batch_size * T - 1, row sum up to 1
                context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size
                if t < self.T :
                    y_tilde = self.decoder_fc(torch.cat((context, Y[:, t].unsqueeze(1)), dim = 1)) # batch_size * 1
                    self.decoder_lstm_layer.flatten_parameters()
                    _, lstm_output = self.decoder_lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                    hidden = lstm_output[0] # 1 * batch_size * decoder_hidden_size
                    cell = lstm_output[1] # 1 * batch_size * decoder_hidden_size
            input_final[:,i,:] = hidden[0]
        
        #mil
        out_fc = self.fc(input_final)
        out_p = self.logitic(out_fc)
        
        return out_p


# In[ ]:




