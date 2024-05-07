import scipy.io
import numpy as np


#hyperparameters 
p = 100


#############################################################################################
#save_path
data_path = '.../data/'
#############################################################################################

#############################################################################################
#domain
lb = np.array([0])
ub = np.array([1])

scipy.io.savemat(data_path+'1D_Boundary.mat', {'lb':lb,'ub':ub})
#############################################################################################



#############################################################################################
#residual points
Train_Grid_Size = 200
X_mesh = np.linspace(lb[0], ub[0],Train_Grid_Size)
x_mesh = X_mesh[1:-1]
X_res = x_mesh.flatten()[:, None]
scipy.io.savemat(data_path+'1D_X_Res.mat', {'X_res':X_res})
#############################################################################################



#############################################################################################
#the analytic solution
def Exact_U(x):
    u = np.exp(-p*(x-0.5)**2)
    return u

#the first-order derivative
def Exact_U_x(x):
    u = (p - 2*p*x)*np.exp(-p*(x-0.5)**2)
    return u

U_res = Exact_U(X_res)
U_x_res = Exact_U_x(X_res)

scipy.io.savemat(data_path+'1D_U_Res.mat', {'U_res':U_res})
scipy.io.savemat(data_path+'1D_U_x_Res.mat', {'U_x_res':U_x_res})
#############################################################################################




#############################################################################################
#for predicting
Train_Grid_Size = 25
X_mesh = np.linspace(lb[0], ub[0],Train_Grid_Size)
x_mesh = X_mesh[1:-1]
X_test = x_mesh.flatten()[:, None]
scipy.io.savemat(data_path+'1D_X_Pred.mat', {'X_pred':X_test})
#############################################################################################



