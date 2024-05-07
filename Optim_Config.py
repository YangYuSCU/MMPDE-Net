"""
@author: yangyu
"""


import torch
from NN_Config import MMPDENet
from Train_Config import TrainConfig
import scipy.io

class OptimConfig():
    def __init__(self,device,data_path):
        super(OptimConfig, self).__init__()   
        
        self.device = device
        
        #Path to load and save data
        self.data_path = data_path
        
        train_dict = TrainConfig(self.device,self.data_path).Train_Dict()
        param_dict = TrainConfig(self.device,self.data_path).Param_Dict()
        
        #Load MMPDENet
        self.model = MMPDENet(train_dict=train_dict, param_dict=param_dict)
        self.model.to(device)
        
        #Random seed
        torch.manual_seed(100)
        torch.cuda.manual_seed_all(100)

    #Optimize and predict
    def OptimAndPredi(self,n_steps_1,n_steps_2):
        Adam_optimizer = torch.optim.Adam(params=self.model.weights + self.model.biases,
                                            lr=1e-3,
                                            betas=(0.9, 0.999),
                                            eps=1e-8,
                                            weight_decay=0,
                                            amsgrad=False)
        self.model.train_Adam(Adam_optimizer, n_steps_1, None)
        
        #############################################################################
        LBFGS_optimizer = torch.optim.LBFGS(
            params=self.model.weights + self.model.biases,
            lr=1,
            max_iter=n_steps_2,
            tolerance_grad=-1,
            tolerance_change=-1,
            history_size=100,
            line_search_fn=None)
        self.model.train_LBFGS(LBFGS_optimizer, None)    
        #############################################################################

        Case1_X_Pred = scipy.io.loadmat(self.data_path + '1D_X_Pred.mat')
        self.X_pred = Case1_X_Pred['X_pred']
        X_new = self.model.predict(self.X_pred) 
        scipy.io.savemat(self.data_path +'1D_X_new.mat', {'X_new':X_new})
        
        
