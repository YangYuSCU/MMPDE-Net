import time
import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable




class MMPDENet(nn.Module):

    def __init__(self, train_dict, param_dict):
        super(MMPDENet, self).__init__()
        
        #Retrieve data
        self.layers, self.data_path,self.device = self.unzip_param_dict(
            param_dict=param_dict)

        lb, ub, X_res, U_res ,U_x_res= self.unzip_train_dict(
            train_dict=train_dict)    




        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)


        self.X_res = self.data_loader(X_res)
        self.U_res = self.data_loader(U_res)
        self.U_x_res = self.data_loader(U_x_res)

        self.dt = 0.1

        self.weights, self.biases = self.initialize_NN(self.layers)

        self.loss = None
        
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        
        self.nIter = 0
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None

        self.start_time = None

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = Variable(torch.zeros([1, layers[l + 1]],
                                     dtype=torch.float32)).to(self.device)
            b.requires_grad_()
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    def detach(self, data):
        return data.detach().cpu().numpy()
    
    def xavier_init(self, size):
        W = Variable(nn.init.xavier_normal_(torch.empty(size[0], size[1]))).to(
            self.device)
        W.requires_grad_()
        return W

    def data_loader(self, x, requires_grad=True):
        x_tensor = torch.tensor(x,
                                requires_grad=requires_grad,
                                dtype=torch.float32)
        return x_tensor.to(self.device)

    def coor_shift(self, X):
        X_shift = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return X_shift

    def unzip_train_dict(self, train_dict):
        train_data = (  train_dict['lb'], 
                        train_dict['ub'], 
                        train_dict['X_res'], 
                        train_dict['U_res'],
                        train_dict['U_x_res'],
        )
        return train_data

    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['layers'],
                      param_dict['data_path'],
                      param_dict['device'])
        return param_data

    def neural_net(self, x, weights, biases):
        num_layers = len(weights) + 1
        X = self.coor_shift(x)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            X = torch.tanh(torch.add(torch.matmul(X, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(X, W), b)  
        return Y

    def net_u(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        g0 = (1 - torch.exp(-(x-self.lb[0])))
        g1 = (1 - torch.exp(-(x-self.ub[0])))
        u = g0*g1*u + x
        return u

    def grad_u(self, u, x):
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x

    def grad_2_u(self, u, x):
        u_x = self.grad_u(u,x)
        u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
        return u_x,u_xx

    def monitor_func(self,U):
        w = (1+100*U)**(1/2)
        return w

    def forward(self, x):
        u= self.net_u(x)
        return self.detach(u).squeeze()


    def loss_func(self, pred_, true_=None, alpha=1):
        if true_ is None:
            true_ = torch.zeros_like(pred_).to(self.device)
        return alpha * self.loss_fn(pred_, true_)



    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        #Loss function initialization
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()

        # Loss_MMPDE_res
        loss_MMPDE = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss_MMPDE.requires_grad_()
        Xi = self.X_res
        x = self.net_u(Xi)
        x_Xi,x_XiXi = self.grad_2_u(x,Xi)

        MMPDE_res = ((x - Xi)/self.dt)*(self.monitor_func(self.U_res)**2)*(x_Xi**2) +\
        (self.monitor_func(self.U_res)*x_XiXi+ 50*(1+100*self.U_res)**(-1/2)*self.U_x_res*x_Xi)

        loss_MMPDE = self.loss_func(MMPDE_res)

            
        # Weights
        alpha_res = 1

        self.loss = loss_MMPDE * alpha_res 
        self.loss.backward()
        self.nIter = self.nIter + 1

        loss_remainder = 10
        if np.remainder(self.nIter, loss_remainder) == 0:
            loss_MMPDE = self.detach(loss_MMPDE)
            loss = self.detach(self.loss)
            
            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' Loss ' + str(loss) +\
                ' loss_MMPDE ' + str(loss_MMPDE) 
            print(log_str)


            elapsed = time.time() - self.start_time
            print('Iter:', loss_remainder, 'Time: %.4f' % (elapsed))

            self.start_time = time.time()
            
        return self.loss

    def train_LBFGS(self, optimizer, LBFGS_scheduler):
        self.optimizer = optimizer
        self.optimizer_name = 'LBFGS'
        self.scheduler = LBFGS_scheduler

        def closure():
            loss = self.optimize_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            return loss

        self.optimizer.step(closure)

    def train_Adam(self, optimizer, nIter, Adam_scheduler):
        self.optimizer = optimizer
        self.optimizer_name = 'Adam'
        self.scheduler = Adam_scheduler
        for it in range(nIter):
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(self.loss)

        

    def predict(self, X_input):
        x = self.data_loader(X_input)
        with torch.no_grad():
            u= self.forward(x)
            return u

