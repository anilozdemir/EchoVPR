# from .evaluate import m_accuracy, evaluate
# import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def getValidationIndices(randSeed, nOutput):
    k_fold   = 10 # number of folds
    N        = nOutput # number of images in total
    randN    = np.arange(N)
    np.random.seed(randSeed) # random seed for reproducibility
    np.random.shuffle(randN) # shuffle the numbers
    TestInd  = randN.reshape(k_fold, -1) # test indices (N/k_fold %), k_fold of them
    TrainInd = np.array([list(set(np.arange(N)).difference(set(tInd))) for tInd in TestInd]) # all data except test indices
    return TrainInd, TestInd


class ESN_Exp():
    def __init__(self,model,hlrTrain,hlrTest,TestInd,ValidationInd,tol):
        
        self.model         = model
        self.hlrTrain      = hlrTrain
        self.hlrTest       = hlrTest
        self.TestInd       = TestInd
        self.ValidationInd = ValidationInd
        self.tol           = tol

        self.hierModel = True if hasattr(model, 'hierModel') else False # check if the model is hierarchical

        self.precalc()
        
    def precalc(self):
        
        if self.model.useReadout == True:
            raise ValueError('This class require ESN.useReadout==False!...')
        
        model    = self.model      
        hlrTrain = self.hlrTrain
        hlrTest  = self.hlrTest  
        
        activation = nn.Softmax(dim=1) 

        if self.hierModel:  # if the model is hierarchical ESN
            with torch.no_grad():
                # get training states
                model.reset()
                [activation(model(torch.Tensor(x).unsqueeze(0))) for x in hlrTrain]
                hs1_train = model.getNumpyData()['hiddenStates1'][1:]
                hs2_train = model.getNumpyData()['hiddenStates2'][1:]
                # get testing states
                model.reset()
                [activation(model(torch.Tensor(x).unsqueeze(0))) for x in hlrTest]
                hs1_test = model.getNumpyData()['hiddenStates1'][1:]
                hs2_test = model.getNumpyData()['hiddenStates2'][1:]
                    
            self.hs_train = torch.Tensor(np.hstack([hs1_train, hs2_train])).to(model.device)
            self.hs_test  = torch.Tensor(np.hstack([hs1_test , hs2_test ])).to(model.device) 
       
        else: # if it is single ESN
            # get states
            with torch.no_grad():
                # get training states
                model.reset()
                [activation(model(torch.Tensor(x).unsqueeze(0))) for x in hlrTrain]
                hs_train = model.getNumpyData()['hiddenStates'][1:]
                # get testing states
                model.reset()
                [activation(model(torch.Tensor(x).unsqueeze(0))) for x in hlrTest]
                hs_test = model.getNumpyData()['hiddenStates'][1:]

            self.hs_train = torch.Tensor(hs_train).to(model.device)
            self.hs_test  = torch.Tensor(hs_test ).to(model.device) 

    def train_esn(self, nEpoch = 50, lR = 0.001, nBatch = 5, returnData=True, returnDataAll = False):
        
        tol           = self.tol
        model         = self.model      
        hs_train      = self.hs_train
        hs_test       = self.hs_test  
        TestInd       = self.TestInd
        ValidationInd = self.ValidationInd 
          
        k_fold         = TestInd.shape[0]
        
        OutputTrain    = np.zeros((nEpoch, model.nOutput, model.nOutput))
        OutputTest_All = np.zeros((nEpoch, model.nOutput, model.nOutput))
        OutputValid    = np.zeros((nEpoch, k_fold, model.nOutput//k_fold, model.nOutput)) # 10 % data
        OutputTest     = np.zeros((nEpoch, k_fold, model.nOutput-model.nOutput//k_fold, model.nOutput)) # 90 % data
           
        Loss           = np.zeros((nEpoch,nBatch))
        AccTrain       = np.zeros((nEpoch))
        AccTest        = np.zeros((nEpoch,k_fold))
        AccValid       = np.zeros((nEpoch,k_fold))
        
        criterion      = torch.nn.CrossEntropyLoss()
        optimizer      = torch.optim.Adam(model.parameters(), lr=lR)
        activation     = torch.nn.Softmax(dim=1) 

        # reshape training and label sets
        batch_train    = hs_train.reshape(nBatch, -1, hs_train.shape[-1]).to(model.device)
        Labels         = np.arange(len(hs_train))
        batch_label    = torch.tensor(Labels).reshape(nBatch,-1).to(model.device)
        hs_test_cpu    = hs_test.to(model.device)
        
        for epoch in range(nEpoch):
            list_out,list_loss = [], []
            for d, x in enumerate(batch_train):
                optimizer.zero_grad()
                out      = activation(F.linear(x, model.Wout))
                loss     = criterion(out, batch_label[d])
                loss.backward()
                optimizer.step()
                list_out.append(out)
                list_loss.append(loss)
            OutputTrain[epoch] = torch.vstack(list_out).detach().cpu()
            Loss[epoch]        = torch.stack(list_loss).detach().cpu()
            AccTrain[epoch]    = np.sum(np.abs(OutputTrain[epoch].argmax(axis=1) - Labels) <= tol)/len(Labels)
            
            with torch.no_grad():
                OutputTest_All[epoch] = activation(F.linear(hs_test_cpu, model.Wout)).detach().cpu()
                for iVal in range(k_fold):
                    OutputValid[epoch, iVal]  = OutputTest_All[epoch][ValidationInd[iVal]]
                    OutputTest[epoch, iVal]   = OutputTest_All[epoch][TestInd[iVal]]
            AccValid[epoch] = np.sum(np.abs(OutputValid[epoch].argmax(axis=2) - ValidationInd) <= tol,axis=1)/ValidationInd.shape[1]
            AccTest[epoch]  = np.sum(np.abs(OutputTest[epoch].argmax(axis=2) - TestInd) <= tol,axis=1)/TestInd.shape[1]
        
        self.Loss           = Loss          
        self.AccTrain       = AccTrain      
        self.AccTest        = AccTest       
        self.AccValid       = AccValid
        
        if returnDataAll:
            self.OutputTrain    = OutputTrain   
            self.OutputTest_All = OutputTest_All
            self.OutputValid    = OutputValid   
            self.OutputTest     = OutputTest  
            return self.getNumpyDataAll()
        
        if returnData:
            return self.getNumpyData()
            
    def train_sparce(self, nEpoch = 50, lR = 0.001, nBatch = 5,  quantile = 0.5, lr_divide_factor = 10, returnData=True, returnDataAll=False):
        
        tol           = self.tol
        model         = self.model      
        hs_train      = self.hs_train
        hs_test       = self.hs_test  
        TestInd       = self.TestInd
        ValidationInd = self.ValidationInd 
          
        k_fold         = TestInd.shape[0]
        
        OutputTrain    = np.zeros((nEpoch, model.nOutput, model.nOutput))
        OutputTest_All = np.zeros((nEpoch, model.nOutput, model.nOutput))
        OutputValid    = np.zeros((nEpoch, k_fold, model.nOutput//k_fold, model.nOutput)) # 10 % data
        OutputTest     = np.zeros((nEpoch, k_fold,model.nOutput-model.nOutput//k_fold, model.nOutput)) # 90 % data
           
        Loss           = np.zeros((nEpoch,nBatch))
        AccTrain       = np.zeros((nEpoch))
        AccTest        = np.zeros((nEpoch,k_fold))
        AccValid       = np.zeros((nEpoch,k_fold))
        
        
        ThrTest_All    = np.zeros((nEpoch, model.nOutput, model.nReservoir))
        ThrTrain_All   = np.zeros((nEpoch, model.nOutput, model.nReservoir))

        # SPARCE MODEL
        threshold_fixed = torch.nn.Parameter(torch.quantile(torch.abs(hs_train), quantile, dim=0).unsqueeze_(0), requires_grad = False)  # update 2 # calculate quantile values (one for each neuron, over nSample samples)
        Thr_learn       = torch.nn.Parameter(torch.zeros((*threshold_fixed.shape)), requires_grad = True).to(model.device)  # update
        
        # SPARCE MODEL UPDATE
        model.quantile = quantile
        model.thrGrad  = True
        model.Thr      = Thr_learn 
        model.lR_Thr   = lR / lr_divide_factor
        
        # SPARCE MODEL OPTIMIZER
        params         = [{'params': model.Wout, 'lr': lR}, {'params':  model.Thr, 'lr': model.lR_Thr}]
        optimizer      = torch.optim.Adam(params)

        # LOSS & REST (the same)
        criterion      = nn.CrossEntropyLoss()
        activation     = nn.Softmax(dim=1) 
        # reshape training and label sets
        batch_train    = hs_train.reshape(nBatch, -1, hs_train.shape[-1]).to(model.device)
        Labels         = np.arange(len(hs_train))
        batch_label    = torch.tensor(Labels).reshape(nBatch,-1).to(model.device)
        hs_test_cpu    = hs_test.to(model.device)
        
        for epoch in range(nEpoch):
            list_out,list_loss,list_thr = [], [],[]
            for d, x in enumerate(batch_train):
                optimizer.zero_grad()
                thr      = torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - model.Thr - threshold_fixed) # SPARCE
                out      = activation(F.linear(thr, model.Wout))                          # update 1: SPARCE
                loss     = criterion(out, batch_label[d])
                list_thr.append(thr)
                loss.backward()
                optimizer.step()
                list_out.append(out)
                list_loss.append(loss)
            OutputTrain[epoch] = torch.vstack(list_out).detach()
            Loss[epoch]        = torch.stack(list_loss).detach()
            AccTrain[epoch]    = np.sum(np.abs(OutputTrain[epoch].argmax(axis=1) - Labels) <= tol)/len(Labels)
            ThrTrain_All[epoch] = torch.vstack(list_thr).detach()
            
            with torch.no_grad():
                threshold_fixed_valid = torch.nn.Parameter(torch.quantile(torch.abs(hs_test), quantile, dim=0).unsqueeze_(0), requires_grad = False)
                threshold_activities  = torch.sign(hs_test_cpu) * torch.nn.functional.relu(torch.abs(hs_test_cpu) - model.Thr - threshold_fixed_valid) # SPARCE (on whole dataset)
                OutputTest_All[epoch] = activation(F.linear(threshold_activities, model.Wout)).detach()
                ThrTest_All[epoch]    = threshold_activities.detach()
                
                for iVal in range(k_fold):
                    OutputValid[epoch, iVal]  = OutputTest_All[epoch][ValidationInd[iVal]]
                    OutputTest[epoch, iVal]   = OutputTest_All[epoch][TestInd[iVal]]
            AccValid[epoch] = np.sum(np.abs(OutputValid[epoch].argmax(axis=2) - ValidationInd) <= tol,axis=1)/ValidationInd.shape[1]
            AccTest[epoch]  = np.sum(np.abs(OutputTest[epoch].argmax(axis=2)  - TestInd)       <= tol,axis=1)/TestInd.shape[1]
        
        self.Loss           = Loss          
        self.AccTrain       = AccTrain      
        self.AccTest        = AccTest       
        self.AccValid       = AccValid
        
        if returnDataAll:
            self.OutputTrain    = OutputTrain   
            self.OutputTest_All = OutputTest_All
            self.OutputValid    = OutputValid   
            self.OutputTest     = OutputTest  
            self.ThrTest_All    = ThrTest_All
            self.ThrTrain_All   = ThrTrain_All
            return self.getNumpyDataAll()
        
        if returnData:
            return self.getNumpyData()
        
    def getNumpyData(self):
        dataStr   = ['Loss', 'AccTrain', 'AccTest', 'AccValid']
        dataNumpy = [self.Loss, self.AccTrain, self.AccTest, self.AccValid]
        return dict(zip(dataStr, dataNumpy))
    
    def getNumpyDataAll(self):
        dataStr   = ['OutputTrain', 'OutputTest_All', 'OutputValid', 'OutputTest', 'Loss', 'AccTrain', 'AccTest', 'AccValid']
        dataNumpy = [self.OutputTrain, self.OutputTest_All, self.OutputValid, self.OutputTest, self.Loss, self.AccTrain, self.AccTest, self.AccValid]
        if self.model.isSPARCE: 
            dataStr   += ['ThrTest_All', 'ThrTrain_All']
            dataNumpy += [self.ThrTest_All, self.ThrTrain_All]
        
        return dict(zip(dataStr, dataNumpy))
    

    # TODO:
    # def plotTrain(self):
    #     fig,ax = P.subplots(figsize=(10,5))
    #     ax.plot(self.Loss.mean(axis=1),c='C3',label='loss');
    #     ax.set_ylabel('loss')
        
    #     ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    #     ax2.plot(self.AccTrain,label='train');
    #     ax2.plot(self.AccTest.mean(axis=1),label='mean test',ls='-.');
    #     ax2.plot(self.AccValid.mean(axis=1),label='mean valid',ls='--',alpha=0.6);
    #     ax2.set_ylabel('accuracy')
    #     ax.legend(loc=1);
    #     ax2.legend();
        
    # def plotHeatmap(self):
    #     def plotD(epoch):
    #         fig, ax = P.subplots(1,2,figsize=(20,9))
    #         im=ax[0].imshow(softmax(OutputTrain[epoch],axis=1),norm=LogNorm())
    #         ax[0].axis('off')
    #         cbar_ax = fig.add_axes([0.16, 0.07, 0.25, 0.05])
    #         cbar=fig.colorbar(im, cax=cbar_ax,orientation="horizontal")
    #         ax[0].set_title(f'train acc: {m_accuracy(OutputTrain[epoch],tol)}')
            
    #         fig.subplots_adjust(right=0.8)
            
    #         ax[1].bar(np.arange(k_fold)-0.2, AccValid[epoch],0.4,label='valid')
    #         ax[1].bar(np.arange(k_fold)+0.2, AccTest[epoch],0.4,label='test')
    #         ax[1].set_xlabel('folds')
    #         ax[1].set_xticks(np.arange(k_fold));
    #         ax[1].set_ylabel('accuracy');
    #         ax[1].legend()
    #         ax[1].set_ylim(0,1.04)
    #         ax[1].set_title(f'mean acc, test: {AccTest[epoch].mean():.2f} valid: {AccValid[epoch].mean():.2f}')
        
    #     epochW = widgets.SelectionSlider(description='epoch:', options=range(nEpoch), value  = 0  , readout=True)  # widget for input noise
    #     outD   = widgets.interactive_output(plotD, {'epoch': epochW})
    #     display(widgets.HBox([epochW]), outD)    
