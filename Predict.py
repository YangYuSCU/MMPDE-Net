import matplotlib.pyplot as plt
import scipy.io



if __name__ == "__main__":
    


    #Path to load and save data
    data_path = '.../data/'
    

    #Load data
    Case1_X_new = scipy.io.loadmat(data_path + '1D_X_new.mat')
    X_new = Case1_X_new['X_new'].flatten()[:, None]

    def Y_aixs_1(x):
        u = 1 + 0*x
        return u   
    
    plt.plot(X_new,Y_aixs_1(X_new),'bo',markersize=4.0)
    
    file_name = data_path+ '/' 
    
    plt.savefig(file_name + 'Points.png')
    plt.show()
