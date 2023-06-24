# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy as sc
import random
from torch.nn.modules import Module
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import sys
def input_state_concatenation(input,state):
    temp_input = torch.cat((input,state),dim=1)
    return temp_input

def state_concatenation_np(input,state):
    temp_input = np.concatenate((input,state),axis=1)
    return temp_input

def state_transform(state,flag = 1):
    if flag == 1:
        state_dim = state[0].shape[1]
        _state = torch.Tensor(len(state),state_dim)
        for col_idx in range(len(state)):
            _state[col_idx,:] = state[col_idx]
    else:
        state_dim = state[0].shape[0]
        _state = torch.Tensor(len(state),state_dim)
        for col_idx in range(len(state)):
            _state[col_idx,:] = state[col_idx]
    return _state


def MAPE(label,predicted):
    result = np.mean(np.abs((label - predicted) / label))
    return result
def SMAPE(label,predicted):
    return 1/len(label) * np.sum(2 * np.abs(predicted - label) / (np.abs(label) + np.abs(predicted)))
    # 100 is 100% 

class ESN(Module):

    def __init__(self,input,reservoir,sr,scale_in,leaking_rate,density,Nepochs,eta,mu,sigma,threshold,
                    Win_assign='Uniform',W_assign = 'Uniform'):
        super(ESN, self).__init__()
        self.input = input
        self.reservoir = reservoir
        self.spectral_radius = sr
        #self.sparsity是self.sparsity*100%的weights为0
        self.density = density
        self.scale_in = scale_in
        self.leaking_rate = leaking_rate
        self.W = None
        self.Win = None
        self.Win_assigned = Win_assign
        self.W_assigned = W_assign
        self.state_list = []
        self.state = None
        self.Nepochs = Nepochs
        self.threshold = threshold
        self.gain = None
        self.bias = None
        self.eta = eta
        self.mu = mu
        self.sigma = sigma
        
        self.reset_parameters()
        
    def reset_parameters(self):
        key_words_for_Win = ['Xvaier','Uniform','Guassian']
        key_words_for_W = ['Xvaier','Uniform','Guassian']
        try:
            key_words_for_Win.index(self.Win_assigned)
        except ValueError:
            raise ValueError("Only Xvaier,Uniform and Guassian types for assignment of Win")
        try:
            key_words_for_W.index(self.W_assigned)
        except ValueError:
            raise ValueError("Only Xvaier,Uniform and Guassian types for assignment of W")
        #np.random.seed(0)
        if self.Win_assigned == 'Uniform':
            Win_np = np.random.uniform(-self.scale_in,self.scale_in,size=(self.input+1,self.reservoir))
        elif self.Win_assigned =='Xvaier':
            Win_np = (np.random.randn(self.input+1,self.reservoir)/np.sqrt(self.input+1))*self.scale_in
        elif self.Win_assigned == 'Guassian':
            Win_np = np.random.randn(self.input+1,self.reservoir)*self.scale_in
        #np.random.seed(0)
        if self.W_assigned == 'Uniform':
            if self.density<1:
                W_np = np.zeros((self.reservoir,self.reservoir))
                for row in range(self.reservoir):
                    number_row_elements = round(self.density * self.reservoir)
                    #random.seed(0)
                    row_elements = random.sample(range(self.reservoir),int(number_row_elements))
                    #np.random.seed(0)
                    W_np[row,row_elements] = np.random.uniform(-1,+1,size=(1,int(number_row_elements)))
            else:
                #np.random.seed(0)
                W_np = np.random.uniform(-1,+1,size=(self.reservoir,self.reservoir))

        elif self.W_assigned == 'Guassian':
            if self.density<1:
                W_np = np.zeros((self.reservoir,self.reservoir))
                for row in range(self.reservoir):
                    number_row_elements = round(self.density * self.reservoir)
                    row_elements = random.sample(range(self.reservoir),int(number_row_elements))
                    W_np[row,row_elements] = np.random.randn(1,int(number_row_elements))
            else:
                W_np = np.random.randn(self.reservoir,self.reservoir)
        elif self.W_assigned == 'Xvaier':
            if self.density<1:
                W_np = np.zeros((self.reservoir,self.reservoir))
                for row in range(self.reservoir):
                    number_row_elements = round(self.density * self.reservoir)
                    row_elements = random.sample(range(self.reservoir),int(number_row_elements))
                    W_np[row,row_elements] = np.random.randn(1,int(number_row_elements))/np.sqrt(self.reservoir)
            else:
                W_np = np.random.randn(self.reservoir,self.reservoir)/np.sqrt(self.reservoir)


        
        Ws = (1-self.leaking_rate) * np.eye(W_np.shape[0], W_np.shape[1]) + self.leaking_rate * W_np
        eig_values = np.linalg.eigvals(Ws)
        actual_sr = np.max(np.absolute(eig_values))
        Ws = (Ws*self.spectral_radius)/actual_sr
        W_np = (Ws-(1.-self.leaking_rate)*np.eye(W_np.shape[0],W_np.shape[1]))/self.leaking_rate
        Gain_np = np.ones((1,self.reservoir))
        Bias_np = np.zeros((1,self.reservoir))

        self.Win = torch.Tensor(Win_np)
        self.W = torch.Tensor(W_np)
        self.gain = torch.Tensor(Gain_np)
        self.bias = torch.Tensor(Bias_np)
        #print(np.max(np.absolute(np.linalg.eigvals(W_np))))




    
    def forward(self,input,h_0=None,useIP = True, IPmode = 'training'):

        if useIP == True and IPmode =='training':
            self.computeIntrinsicPlasticity(input,useIP = useIP,IPmode = IPmode)
        if useIP == True and IPmode == 'testing':
            return self.computeState(input,h_0 = h_0,useIP = useIP,IPmode = IPmode)
        if useIP == False:
            return self.computeState(input,h_0=h_0,useIP = useIP,IPmode=IPmode)

    def reset_state(self):
        self.state_list = []

    def computeIntrinsicPlasticity(self,input,useIP,IPmode):
        for epoch in range(self.Nepochs):
            Gain_epoch = self.gain
            Bias_epoch = self.bias
            sys.stdout.write("IP training {} epoch.\n".format(epoch+1))
            self.computeState(input,h_0 = None, useIP = True, IPmode = 'training')

            if (torch.norm(self.gain-Gain_epoch,2) < self.threshold) and (torch.norm(self.bias-Bias_epoch,2)< self.threshold):
                sys.stdout.write("IP training training over.\n")
                sys.stdout.flush()
                break
            if epoch+1 == self.Nepochs:
                sys.stdout.write("total IP training epochs are over.\n".format(epoch+1))
                sys.stdout.write('.')
                sys.stdout.flush()

    
    
    def computeState(self,input,h_0 = None, useIP = False, IPmode = 'testing'):
        back_state_list = []
        if h_0 is None:
            h_0 = torch.zeros(1,self.reservoir)
        state = h_0
        input_bias = torch.ones(input.shape[0],1)
        input = torch.cat((input,input_bias),dim=1)
        if useIP ==True and IPmode =='training':
            state_before_activated = torch.zeros(1,self.reservoir)
            for col_idx in range(input.shape[0]):
                if col_idx == 0:
                    e = input[col_idx,:].unsqueeze(0)
                    input_part = torch.mm(e,self.Win)
                    inner_part = torch.mm(state,self.W)
                    state_before_activated = input_part+inner_part
                    state = (1.0-self.leaking_rate)*state+(self.leaking_rate)*torch.tanh(self.gain*state_before_activated+self.bias)
                else:
                    e = input[col_idx,:].unsqueeze(0)
                    input_part = torch.mm(e,self.Win)
                    inner_part = torch.mm(state,self.W)
                    state_before_activated = input_part+inner_part
                    state = (1.0-self.leaking_rate)*state+(self.leaking_rate)*torch.tanh(self.gain * state_before_activated+self.bias)
                    eta = self.eta
                    mu = self.mu
                    sigma2 = self.sigma**2
                    deltaBias = -eta*((-mu/sigma2)+state*(2*sigma2+1-(state**2)+mu*state)/sigma2)
                    deltaGain = eta/self.gain + deltaBias*state_before_activated
                    self.gain = self.gain+deltaGain
                    self.bias = self.bias+deltaBias
                self.state_list.append(state)
            back_state_list = self.state_list
            self.reset_state()
        if useIP == True and IPmode == 'testing':
            state_before_activated = torch.zeros(1,self.reservoir)
            for col_idx in range(input.shape[0]):
                e = input[col_idx,:].unsqueeze(0)
                input_part = torch.mm(e,self.Win)
                inner_part = torch.mm(state,self.W)
                state_before_activated = input_part+inner_part
                state = (1.0-self.leaking_rate)*state+(self.leaking_rate)*torch.tanh(self.gain * state_before_activated+self.bias)      
                self.state_list.append(state)
            back_state_list = self.state_list
            self.reset_state()
        if useIP == False:
            state_before_activated = torch.zeros(1,self.reservoir)
            for col_idx in range(input.shape[0]):
                e = input[col_idx,:].unsqueeze(0)
                input_part = torch.mm(e,self.Win)
                inner_part = torch.mm(state,self.W)
                state_before_activated = input_part+inner_part
                state = (1.0-self.leaking_rate)*state+(self.leaking_rate)*torch.tanh(state_before_activated)
                self.state_list.append(state)
            back_state_list = self.state_list
            self.reset_state()
        return back_state_list


''' if __name__ == "__main__":
    #创建一个esn，这个esn里面的参数就是下面这些
    layer1 = ESN(input = 1,reservoir = 1000,output = 1,sr = 0.99,sparsity = 0.9,scale_in = 1,scale_bias = 0.01,leaking_rate =0.97,W_assign='Uniform',Win_assign='Xvaier')
    data_source = pd.read_csv('MackeyGlass_t17.txt',header=None)
    DF = torch.Tensor(pd.DataFrame(data_source).values)
    totalTrain = DF[:-1,:]
    totalTest = DF[1:,:]
    num_train = 5000
    washout = 500
    loss_function = nn.MSELoss()
    trX = totalTrain[:num_train,:]
    trY = totalTest[:num_train,:]
    teX = totalTrain[num_train:,:]
    teY = totalTest[num_train:,:]
    #训练阶段，h_0 = None，代表state(t=0) = 零矩阵
    train_state_list = layer1(trX,h_0 = None)
    #将list的state转化成矩阵。
    train_state = state_transform(train_state_list)
    #将state矩阵和input进行连接，当然也可以不进行这一步（不调用input_state_concatenation函数)
    #train_state = input_state_concatenation(trX,train_state)
    #洗掉washout部分，state和label对应
    train_state = train_state[washout:,:]
    trY = trY[washout:,:]
    train_state_np = train_state.detach().numpy()
    trY_np = trY.numpy()
    #创建线性回归，如果punish_factor不等于0，就是脊回归，否则就是普通的线性回归。当esn维度上升后，用脊回归效果好
    Ridge_1 = Ridge(alpha=1e-16,fit_intercept=True,copy_X=True).fit(train_state_np,trY_np)
    print(mean_squared_error(Ridge_1.predict(train_state_np),trY_np))
    test_state_list = layer1(teX,h_0 = train_state_list[-1])
    test_state = state_transform(test_state_list)
    test_state_np = test_state.detach().numpy()
    teY_np = teY.numpy()
    print(mean_squared_error(Ridge_1.predict(test_state_np),teY_np))
    print(Ridge_1.get_params()) '''

if __name__ == "__main__":
    #创建一个esn，这个esn里面的参数就是下面这些
    #torch.manual_seed(0)
    #np.random.seed(0)
    layer1 = ESN(input = 1,reservoir = 100,sr = 0.95,density = 0.1,scale_in = 0.1,leaking_rate =1.0,Nepochs=10,eta=5e-4,mu=0,sigma=0.1,threshold=0.1,W_assign='Uniform',Win_assign='Uniform')
    data_source = pd.read_csv('mackey_glass_t17_original.txt',header=None)
    DF = torch.Tensor(pd.DataFrame(data_source).values)
    totalTrain = DF[:-1,:]
    totalTest = DF[1:,:]
    num_train = 5000
    washout = 500
    loss_function = nn.MSELoss()
    trX = totalTrain[:num_train,:]
    trY = totalTest[:num_train,:]
    teX = totalTrain[num_train:,:]
    teY = totalTest[num_train:,:]
    #训练阶段，h_0 = None，代表state(t=0) = 零矩阵
    layer1(trX,h_0 = None,useIP = True, IPmode = 'training')
    train_state_list = layer1(trX,h_0 = None,useIP = True, IPmode = 'testing')
    #将list的state转化成矩阵。
    train_state = state_transform(train_state_list)
    #将state矩阵和input进行连接，当然也可以不进行这一步（不调用input_state_concatenation函数)
    #train_state = input_state_concatenation(trX,train_state)
    #洗掉washout部分，state和label对应
    train_state = train_state[washout:,:]
    trY_washout = trY[washout:,:]
    train_state_np = train_state.detach().numpy()
    trY_np = trY_washout.numpy()
    #创建线性回归，如果punish_factor不等于0，就是脊回归，否则就是普通的线性回归。当esn维度上升后，用脊回归效果好
    Ridge_1 = Ridge(fit_intercept=True,copy_X=True,solver="auto").fit(train_state_np,trY_np)
    print(mean_squared_error(Ridge_1.predict(train_state_np),trY_np))
    test_state_list = layer1(teX,h_0 = train_state_list[-1],useIP=True,IPmode='testing')
    test_state = state_transform(test_state_list)
    test_state_np = test_state.detach().numpy()
    teY_np = teY.numpy()
    print(mean_squared_error(Ridge_1.predict(test_state_np),teY_np))

    train_state_list = layer1(trX,h_0 = None,useIP = False)
    train_state = state_transform(train_state_list)
    train_state = train_state[washout:,:]
    trY_washout = trY[washout:,:]
    train_state_np = train_state.detach().numpy()
    trY_np = trY_washout.numpy()
    Ridge_1 = Ridge(fit_intercept=True,copy_X=True,solver="auto").fit(train_state_np,trY_np)
    print(mean_squared_error(Ridge_1.predict(train_state_np),trY_np))
    test_state_list = layer1(teX,h_0 = train_state_list[-1],useIP=False)
    test_state = state_transform(test_state_list)
    test_state_np = test_state.detach().numpy()
    teY_np = teY.numpy()
    print(mean_squared_error(Ridge_1.predict(test_state_np),teY_np))
    





    
    
    


    






    


    






        



        

        
        
        

    