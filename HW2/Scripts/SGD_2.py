import numpy as np 
class SGD:
    def __init__(self,X_tr,ytr,X_te,yte,n=[100,200,300],epsilon=[0.001,0.1,1],epochs=[2,3,4],alpha=[2,3,4]) -> None:

        self.X_tr = self.add_bias(X_tr).T #adding bias to training labels , and transposing to reflect the theory
        self.ytr = ytr 
        self.X_te = self.add_bias(X_te).T #adding bias to testing labels , and transposing to reflect the theory
        self.yte = yte 
        self.H = np.array(np.meshgrid(n,epsilon,epochs,alpha)).T.reshape(-1,4) #creating combination of all the hyper parameters 
        pass
    def add_bias(self,X):
        #adding a bias term for the Train and test labels 
        return np.hstack((X,np.ones((X.shape[0],1))))

    def split_train_validation(self,split):
        # Randomize the train data 
        allIdxs = np.arange(self.X_tr.shape[1]) 
        Idxs = np.random.permutation(allIdxs) #random indices for the train data 
        # select the 1st 80 % of the indices for thr train data and 20% for validation 
        train_part = Idxs[:int(len(Idxs)*split)]
        validation_part = Idxs[int(len(Idxs)*split):]
        X_tr = self.X_tr[:,train_part]
        ytr = self.ytr[train_part]
        X_va = self.X_tr[:,validation_part]
        yva = self.ytr[validation_part]
        return X_tr,ytr,X_va,yva 

    def gradient(self,X,y,w,alpha):
        alpha = np.full((w.shape[0],1),alpha) 
        alpha[-1] = 0 # set the last value to zero , this represents no regularization for bias 
        # print(w)

        return alpha*w
    def stochastic_gradient_descent(self):
        X_tr,ytr,X_va,yva = self.split_train_validation(0.8) # Split 80% 
        print(self.H)
        h_star =self.H[np.random.choice(len(self.H))] # Initially taking a random hyper parameter set as the best one 
        acc = 0 #initial accuracy  
        for h in self.H: # for each hyper parameter set 
            n,epsilon,epochs ,alpha =h
            print(f'Using hyper parameters Batch Size = {n}, Epsilon = {epsilon}, epochs = {epochs}, alpha = {alpha}')
             
            w = np.random.uniform( size=(self.X_tr.shape[0],1))#initialize the weights with bias term 
            
            n = int(n) 
            epochs = int(epochs) 
            for epoch in range(int(epochs)):
                print(f"Epoch number {epoch}")
                n_ = 0 
                while n_ < len(ytr):
                    X = X_tr[:,n_:n+n_]
                    y = ytr[n_:n+n_]
                    # Compute gradient on this batch 
                    print(self.gradient(X,y,w,alpha))
                    # print(X_tr[:,n_:n+n_],ytr[n_:n+n_])

                    n_+=n

def main():
    #loading the data
    X_tr = np.reshape(np.load("../../HW1/Data/age_regression_Xtr.npy"), (-1, 48*48)) #5000X2304 
    ytr = np.load("../../HW1/Data/age_regression_ytr.npy") #5000 X 1 
    X_te = np.reshape(np.load("../../HW1/Data/age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("../../HW1/Data/age_regression_yte.npy")
    sgd =  SGD(X_tr,ytr,X_te,yte)
    # print(sgd.split_train_validation(0.8)[0].shape)
    sgd.stochastic_gradient_descent()




if __name__ == "__main__":
    main()
