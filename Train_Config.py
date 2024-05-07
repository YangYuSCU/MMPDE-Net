import scipy.io




class TrainConfig():
    def __init__(self,device,data_path):
        super(TrainConfig, self).__init__()
        
        
        self.device  = device

        #Path to load and save data
        self.data_path = data_path
        
        #Training data
        Case1_X_Res = scipy.io.loadmat(self.data_path + '1D_X_Res.mat')
        self.X_res = Case1_X_Res['X_res']

        Case1_U_Res = scipy.io.loadmat(self.data_path + '1D_U_Res.mat')
        self.U_res = Case1_U_Res['U_res']

        Case1_U_x_Res = scipy.io.loadmat(self.data_path + '1D_U_x_Res.mat')
        self.U_x_res = Case1_U_x_Res['U_x_res']        
        
        #Domain boundary
        Case1_Boundary = scipy.io.loadmat(self.data_path + '1D_Boundary.mat')
        self.lb = Case1_Boundary['lb']
        self.ub = Case1_Boundary['ub']

        #Net size
        self.layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        
    def Train_Dict(self):
        return  {'lb':self.lb, 
            'ub':self.ub,
            'X_res': self.X_res,
            'U_res': self.U_res,
            'U_x_res': self.U_x_res}
    
    def Param_Dict(self):
        return  {'layers': self.layers,
           'data_path':self.data_path,
           'device':self.device}
    

    
    



