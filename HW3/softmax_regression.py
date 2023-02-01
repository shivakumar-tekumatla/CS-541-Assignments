import numpy as np 

class SMR:
    def __init__(self,X_tr,ytr,X_te,yte,c =10) -> None:
        self.c = c #classification 
        self.X_tr = self.add_bias(X_tr).T #adding bias to training labels , and transposing to reflect the theory
        self.ytr = self.create_labels(ytr)
        self.X_te = self.add_bias(X_te).T #adding bias to testing labels , and transposing to reflect the theory
        self.yte = self.create_labels(yte) 

        print(self.X_tr.shape)
        print(self.ytr.shape)
        print(self.X_te.shape)
        print(self.yte.shape)
        
        pass
    def add_bias(self,X):
        #adding a bias term for the Train and test labels 
        return np.hstack((X,np.ones((X.shape[0],1))))

    def create_labels(self,y):
        out = np.zeros((y.shape[0],self.c)) #create array of zeros of size of training labels and classes
        for i,val in enumerate(y):
            out[i][val]=1 #set a corresponding class index to 1 
        return out 

    def softmax(self,z):
        exp = np.exp(z) 
        return exp /np.sum(exp)

    def gradient(self,X,y,w):
        z = X.T@w 
        y_tilde = self.softmax(z) 
        n = y.shape[0]
        return (1/n)*X@(y_tilde-y)
    def CE_loss(self,y_tilde,y):
        # log = np.log(y_tilde)
        return -np.sum(y*np.log(y_tilde))

def main():
    X_tr = np.reshape(np.load("Data/fashion_mnist_train_images.npy"), (-1, 28*28)) 
    ytr = np.load("Data/fashion_mnist_train_labels.npy") 
    X_te = np.reshape(np.load("Data/fashion_mnist_test_images.npy"), (-1, 28*28))
    yte = np.load("Data/fashion_mnist_test_labels.npy")

    smr = SMR(X_tr,ytr,X_te,yte,c=10) 



if __name__=="__main__":
    main()


