import random 
import copy
import numpy as np
import torch
import torch.nn as nn

getSparsity = lambda nFixed, nReservoir: 1 - nFixed/nReservoir # the amount of zeros, percentage sparsity

def createSparseMatrix(nReservoir, sparsity):
    '''
    Utility function: creates Sparse Matrix
    Returns:
            W (np.array): sparse matrix of size (**nReservoir**, **nReservoir**).
    '''
    rows, cols = nReservoir, nReservoir
    W = np.random.uniform(-1, 1, (rows, cols)) # randomly chosen matrix, entries in range [-1,1]
    num_zeros = np.ceil(sparsity * rows).astype(np.int) # number of zeros to set
    for iCol in range(cols):
        row_indices  = np.random.permutation(rows) # choose random row indicies
        zero_indices = row_indices[:num_zeros]     # get the num_zeros of them
        W[zero_indices, iCol] = 0                  # set (zero_indicies, iCol) to 0
    return W

class singleESN(nn.Module):
    ''' 
    Simple ESN Reservoir implementation (no readout yet).
    Compulsory arguments
    --------------------
    - nInput     : int
    - nOutput    : int
    - nReservoir : int
    Optional arguments
    ------------------
    - alpha       : 0.5
    - gamma       : 0.01 
    - rho         : 1.0
    - sparsity    : 0.9
    - phi         : 1.0 
    - randomSeed  : 1
    - activation  : torch.tanh 
    - distr       : 'uniform'
    - useReadout  : False
    '''
    def __init__(self, nInput: int, nOutput: int, nReservoir: int, **kwargs):
        super().__init__()
                
        self.nInput      = nInput   
        self.nOutput     = nOutput  
        self.nReservoir  = nReservoir
        self.alpha       = kwargs.get('alpha'     , 0.5)
        self.gamma       = kwargs.get('gamma'     , 0.01)
        self.randomSeed  = kwargs.get('randomSeed', 1)
        self.phi         = kwargs.get('phi'       , 1.0)
        self.rho         = kwargs.get('rho'       , 1.0)
        self.sparsity    = kwargs.get('sparsity'  , 0.9)        
        self.activation  = kwargs.get('activation', torch.tanh)    
        self.distr       = kwargs.get('Wout_distr', 'uniform')
        self.useReadout  = kwargs.get('useReadout', False)
        deviceStr        = kwargs.get('device'    , 'optional')
        
        if deviceStr == 'optional':
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        elif deviceStr.lower() == 'cpu':
            self.device = 'cpu'
        elif deviceStr.lower() in ['cuda', 'gpu']:
            self.device = 'cuda'
        else:
            raise SystemExit('>> Error in device!')
            
        self.dtype  = torch.float
        np.random.seed(self.randomSeed)
        torch.manual_seed(self.randomSeed)
        
        self.Win = torch.tensor(self.gamma * np.random.randn(self.nInput, self.nReservoir), dtype = self.dtype).to(self.device)
        W        = createSparseMatrix(self.nReservoir, self.sparsity)
        self.W   = torch.tensor(self.rho * W / (np.max(np.absolute(np.linalg.eigvals(W)))), dtype = self.dtype).to(self.device)
        
        
        if self.distr == 'uniform':
            wout = np.random.uniform(-self.phi, self.phi, [self.nOutput, self.nReservoir]) / self.nReservoir
        elif self.distr.lower() == 'normal' or  self.distr.lower() == 'gaussian':
            wout = np.random.normal(0, self.phi, [self.nOutput, self.nReservoir]) / self.nReservoir
        else:
            raise SystemExit('>> Error in init. distribution!')

        Wout = torch.tensor(wout, dtype = self.dtype).to(self.device)
        self.Wout = torch.nn.Parameter(Wout, requires_grad = True)

        self.reset()
    


    def leakyIF(self, recursiveState, u):
        vPrev = copy.copy(recursiveState[-1]).unsqueeze(0)
        v = (1 - self.alpha) * vPrev + self.alpha * self.activation(torch.matmul(vPrev, self.W) + torch.matmul(u, self.Win))
        recursiveStateUpdated = torch.cat((recursiveState, v), dim = 0)
        return recursiveStateUpdated, v
        
    def update_leakyIF(self, u):
        self.hiddenStates, x = self.leakyIF(self.hiddenStates, u.flatten()) # leakyIF with input u and hiddenStates (recurrence layer)
        if self.useReadout : x = nn.functional.linear(x, self.Wout)
        return x

    def forward(self, x):
        '''
        x.dims: 
            - dim 0: batch size
            - dim 1: input size
        '''
        if not len(x.shape) == 2:
            raise ValueError('Wrong input format! Shape should be [1,N]')
        x = x.to(self.device)
        x = torch.vstack([self.update_leakyIF(xb) for xb in x]) # for batch operation
        return x
    
    def reset(self):
        self.hiddenStates = torch.zeros([1,self.nReservoir], dtype = torch.float).to(self.device)

    def print_hparams(self, isSPARCE = False):
        
        import pprint
        
        self.hparams = {
                        'nInput'      : self.nInput     , 
                        'nOutput'     : self.nOutput    , 
                        'nReservoir'  : self.nReservoir ,  
                        'alpha'       : self.alpha      , 
                        'gamma'       : self.gamma      , 
                        'rho'         : self.rho        , 
                        'phi'         : self.phi        , 
                        'Wout_distr'  : self.distr      , 
                        'sparsity'    : self.sparsity   , 
                        'randomSeed'  : self.randomSeed ,
                        'useReadout'  : self.useReadout ,
                        'activation'  : self.activation.__name__,
                        'device'      : self.device     }

        if isSPARCE:
            self.hparams['quantile']  = self.quantile
            self.hparams['thrGrad']   = self.thrGrad 
            if hasattr(self, 'lR_Thr'):  self.hparams['lR_Thr'] = self.lR_Thr 
            
        pp = pprint.PrettyPrinter(indent=1)
        pp.pprint(self.hparams)

    def getNumpyData(self, isSPARCE = False):
        dataStr   = ['Win', 'W', 'Wout', 'hiddenStates']
        dataTorch = [self.Win, self.W, self.Wout, self.hiddenStates]
        if isSPARCE:
            dataStr   += ['Thr',]
            dataTorch += [self.Thr,]
            
        dataNumpy = list(map(lambda x: x.detach().cpu().numpy(), dataTorch))
        return dict(zip(dataStr, dataNumpy))
    


class hierESN(nn.Module):
    ''' 
    Hierachical ESN implementation
    Compulsory arguments
    --------------------
    - nInput      : int
    - nOutput     : int
    - nReservoir1 : int
    - nReservoir2 : int
    Optional arguments
    ------------------
    - alpha1       : 0.5
    - alpha2       : 0.5
    - gamma1       : 0.01 
    - gamma2       : 0.01 
    - rho1         : 1.0
    - rho2         : 1.0
    - sparsity1    : 0.9
    - sparsity2    : 0.9
    - activation1  : torch.tanh 
    - activation2  : torch.tanh 
    - randomSeed   : 1
    - phi          : 1.0 
    - distr        : 'uniform'
    - useReadout   : False
    '''
    def __init__(self, nInput: int, nOutput: int, nReservoir1: int, nReservoir2: int, **kwargs):
        super().__init__()
                
        self.nInput       = nInput   
        self.nOutput      = nOutput  
        self.nReservoir1  = nReservoir1
        self.nReservoir2  = nReservoir2
        self.alpha1       = kwargs.get('alpha1'     , 0.5)
        self.alpha2       = kwargs.get('alpha2'     , 0.5)
        self.gamma1       = kwargs.get('gamma1'     , 0.01)
        self.gamma2       = kwargs.get('gamma2'     , 0.01)
        self.rho1         = kwargs.get('rho1'       , 1.0)
        self.rho2         = kwargs.get('rho2'       , 1.0)
        self.sparsity1    = kwargs.get('sparsity1'  , 0.9)        
        self.sparsity2    = kwargs.get('sparsity2'  , 0.9)        
        self.activation1  = kwargs.get('activation1', torch.tanh)    
        self.activation2  = kwargs.get('activation2', torch.tanh)    
        self.phi          = kwargs.get('phi'       , 1.0)
        self.randomSeed   = kwargs.get('randomSeed', 1)
        self.distr        = kwargs.get('Wout_distr', 'uniform')
        self.useReadout   = kwargs.get('useReadout', False)
        deviceStr         = kwargs.get('device'    , 'optional')
        
        if deviceStr == 'optional':
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        elif deviceStr.lower() == 'cpu':
            self.device = 'cpu'
        elif deviceStr.lower() in ['cuda', 'gpu']:
            self.device = 'cuda'
        else:
            raise SystemExit('>> Error in device!')
            
        self.dtype  = torch.float
        np.random.seed(self.randomSeed)
        torch.manual_seed(self.randomSeed)
        
        self.Win1 = torch.tensor(self.gamma1 * np.random.randn(self.nInput, self.nReservoir1), dtype = self.dtype).to(self.device)
        W1        = createSparseMatrix(self.nReservoir1, self.sparsity1)
        self.W1   = torch.tensor(self.rho1 * W1 / (np.max(np.absolute(np.linalg.eigvals(W1)))), dtype = self.dtype).to(self.device)
        
                
        self.Win2 = torch.tensor(self.gamma2 * np.random.randn(self.nReservoir1, self.nReservoir2), dtype = self.dtype).to(self.device)
        W2        = createSparseMatrix(self.nReservoir2, self.sparsity2)
        self.W2   = torch.tensor(self.rho2 * W2 / (np.max(np.absolute(np.linalg.eigvals(W2)))), dtype = self.dtype).to(self.device)
        

        self.nReservoir = self.nReservoir1 + self.nReservoir2
        if self.distr == 'uniform':
            wout = np.random.uniform(-self.phi, self.phi, [self.nOutput, self.nReservoir]) / self.nReservoir
        elif self.distr.lower() == 'normal' or  self.distr.lower() == 'gaussian':
            wout = np.random.normal(0, self.phi, [self.nOutput, self.nReservoir]) / self.nReservoir
        else:
            raise SystemExit('>> Error in init. distribution!')
        Wout = torch.tensor(wout, dtype = self.dtype).to(self.device)
        self.Wout = nn.Parameter(Wout, requires_grad = True)

        self.hierModel = True
        self.reset()
    
    def leakyIF1(self, recursiveState, u):
        vPrev = copy.copy(recursiveState[-1]).unsqueeze(0)
        v = (1 - self.alpha1) * vPrev + self.alpha1 * self.activation1(torch.matmul(vPrev, self.W1) + torch.matmul(u, self.Win1))
        recursiveStateUpdated = torch.cat((recursiveState, v), dim = 0)
        return recursiveStateUpdated, v

    def leakyIF2(self, recursiveState, u):
        vPrev = copy.copy(recursiveState[-1]).unsqueeze(0)
        v = (1 - self.alpha2) * vPrev + self.alpha2 * self.activation2(torch.matmul(vPrev, self.W2) + torch.matmul(u, self.Win2))
        recursiveStateUpdated = torch.cat((recursiveState, v), dim = 0)
        return recursiveStateUpdated, v

    def update_leakyIF(self, u):
        self.hiddenStates1, x1 = self.leakyIF1(self.hiddenStates1, u.flatten())
        self.hiddenStates2, x2 = self.leakyIF2(self.hiddenStates2, x1.flatten())
        
        x = torch.cat([x1,x2], dim=1) # concat 2 hiddenStates
        
        if self.useReadout : x = nn.functional.linear(x, self.Wout)
        
        return x

    def forward(self, x):
        '''
        x.dims: 
            - dim 0: batch size
            - dim 1: input size
        '''
        if not len(x.shape) == 2:
            raise ValueError('Wrong input format! Shape should be [1,N]')
        x = x.to(self.device)
        x = torch.vstack([self.update_leakyIF(xb) for xb in x]) # for batch operation
        return x
    

    def reset(self):
        self.hiddenStates1 = torch.zeros([1,self.nReservoir1], dtype = torch.float).to(self.device)
        self.hiddenStates2 = torch.zeros([1,self.nReservoir2], dtype = torch.float).to(self.device)

    def print_hparams(self, isSPARCE = False):
        
        import pprint
        
        self.hparams = {
                        'nInput'      : self.nInput      , 
                        'nOutput'     : self.nOutput     , 
                        'nReservoir1' : self.nReservoir1 ,  
                        'nReservoir2' : self.nReservoir2 ,  
                        'alpha1'      : self.alpha1      , 
                        'alpha2'      : self.alpha2      , 
                        'gamma1'      : self.gamma1      , 
                        'gamma2'      : self.gamma2      , 
                        'rho1'        : self.rho1        , 
                        'rho2'        : self.rho2        , 
                        'sparsity1'   : self.sparsity1   , 
                        'sparsity2'   : self.sparsity2   , 
                        'randomSeed'  : self.randomSeed  ,
                        'phi'         : self.phi         , 
                        'Wout_distr'  : self.distr       , 
                        'useReadout'  : self.useReadout  ,
                        'activation1' : self.activation1.__name__,
                        'activation2' : self.activation2.__name__,
                        'device'      : self.device      ,   
                        'hierModel'   : self.hierModel   }

        if isSPARCE:
            self.hparams['quantile']  = self.quantile
            self.hparams['thrGrad']   = self.thrGrad 
            if hasattr(self, 'lR_Thr'):  self.hparams['lR_Thr'] = self.lR_Thr 
            
        pp = pprint.PrettyPrinter(indent=1)
        pp.pprint(self.hparams)

    def getNumpyData(self, isSPARCE = False):
        dataStr   = ['Win1', 'Win2', 'W1', 'W2', 'Wout', 'hiddenStates1', 'hiddenStates2']
        dataTorch = [self.Win1, self.Win2, self.W1, self.W2, self.Wout, self.hiddenStates1, self.hiddenStates2]
        if isSPARCE:
            dataStr   += ['Thr',]
            dataTorch += [self.Thr,]
            
        dataNumpy = list(map(lambda x: x.detach().cpu().numpy(), dataTorch))
        return dict(zip(dataStr, dataNumpy))
    

