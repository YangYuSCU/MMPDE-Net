"""Implementation of MMPDE-Net in Python 
[MMPDE-Net](https://arxiv.org/abs/2311.16167 of our article in arXiv, GitHub: https://github.com/YangYuSCU/MMPDE-Net) (Moving Sampling Physics-informed Neural 
Networks induced by Moving Mesh PDE) In this work, we propose an end-to-end adaptive sampling framework based on deep neural networks and the moving mesh 
method  (MMPDE-Net), which can adaptively generate new sampling points by solving the moving mesh PDE. This model focuses on improving the quality of
sampling points generation. Moreover, we develop an iterative algorithm based on MMPDE-Net, which makes sampling points distribute more precisely and 
controllably. Since MMPDE-Net is independent of the deep learning solver, we combine it with physics-informed neural networks (PINN) to propose moving 
sampling PINN (MS-PINN) and show the error estimate of our method under some assumptions. Finally, we demonstrate the performance improvement of MS-PINN 
compared to PINN through numerical experiments of four typical examples, which numerically verify the effectiveness of our method.

All the computations are carried on NVIDIA A100(80G). 

Author: Yu Yang(yangyu1@stu.scu.edu.cn), Qihong Yang(yangqh0808@163.com), Yangtao Deng(ytdeng1998@foxmail.com), Qiaolin He*(qlhejenny@scu.edu.cn).
Yu Yang, Qihong Yang, Yangtao Deng and Qiaolin He are with School of Mathematics, Sichuan University.

Code author: Yu Yang.
Reviewed by: Qiaolin He.


Copyright
---------
MMPDE-Net is developed in School of Mathematics, Sichuan University (Yu Yang, Qihong Yang, Yangtao Deng, Qiaolin He)
More information about the technique can be found through corresponding authors: Qiaolin He.
"""
import torch
import time
from Optim_Config import OptimConfig


if __name__ == "__main__":
    
    #Use cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('using device ' + device)
    device = torch.device(device)
    
    #Path to load and save data
    data_path = '.../data/'

    
    #Set training Epoches for Adam
    N_Adam = 3000
    #Set training Epoches for LBFGS
    N_LBFGS = 1000
    start_time = time.time()
    OptimConfig(device,data_path).OptimAndPredi(N_Adam,N_LBFGS)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    
    
    