import numpy as np 
class SGD:
    def __init__(self,X_tr,ytr,X_te,yte,n=[10,100,200,300],epsilon=[0.1,0.01,0.000001,0.000001,0.00000001],epochs=[2,3,4,5],alpha=[1,2,3,4],validation_split =0.8) -> None:
        self.X_tr = self.add_bias(X_tr).T #adding bias to training labels , and transposing to reflect the theory
        self.ytr = np.expand_dims(ytr, axis=1)
        self.X_te = self.add_bias(X_te).T #adding bias to testing labels , and transposing to reflect the theory
        self.yte = np.expand_dims(yte, axis=1)
        self.H = np.array(np.meshgrid(n,epsilon,epochs,alpha)).T.reshape(-1,4) #creating combination of all the hyper parameters 
        self.validation_split = validation_split
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
        n = y.shape[0]
        return (X@(X.T@w-y)  + (alpha*w) /n)

    def fMSE(self,X,y,w,alpha):
        reg = alpha*w.T@w -alpha #no regularization for bias 
        return (np.sum((X.T@w-y)**2)+reg)/(2*X.shape[0])

    def train_batch(self,X,y,w,alpha,epsilon,n,n_):
        # Split the batch 
        X = X[:,n_:n+n_]
        y = y[n_:n+n_]
        # Compute gradient on this batch 
        gradient = self.gradient(X,y,w,alpha)
        # Update weights 
        w =  w - epsilon*gradient 
        # Increment the batch start point 
        n_+=n
        return w, n_
    def stochastic_gradient_descent(self):
        X_tr,ytr,X_va,yva = self.split_train_validation(self.validation_split) # Split train to train and validation  
        # print(self.H)
        h_star =self.H[np.random.choice(len(self.H))] # Initially taking a random hyper parameter set as the best one 
        err = np.inf #initial error  
        for h in self.H: # for each hyper parameter set 
            n,epsilon,epochs ,alpha =h
            print(f'Using hyper parameters Batch Size = {n}, Epsilon = {epsilon}, epochs = {epochs}, alpha = {alpha}')
            w = np.random.uniform( size=(self.X_tr.shape[0],1))#initialize the weights with bias term  #np.zeros((self.X_tr.shape[0],1))#
            n = int(n) 
            epochs = int(epochs) 
            for epoch in range(epochs):
                n_ = 0 
                while n_ < len(ytr):
                    w,n_ = self.train_batch(X_tr,ytr,w,alpha,epsilon,n,n_) #Train on the batch
            # test on validation data set 
            curr_err = self.fMSE(X_va,yva,w,alpha)
            print("Error: ", curr_err) #, h)
            if curr_err <err:
                h_star = h  #storing as the best hyper parameter set 
                err = curr_err  

        # Now we found the best hyper parameter set from the given data 
        # Again train on the whole data 
        n,epsilon,epochs ,alpha =h_star #best hyper parameters 
        w = np.zeros((self.X_tr.shape[0],1))#np.random.uniform( size=(self.X_tr.shape[0],1))#initialize the weights with bias term 
        n = int(n) 
        epochs = int(epochs)
        print("The actual training starts....!")
        for epoch in range(int(epochs)):
            # print(f"Epoch number {epoch}")
            n_ = 0 
            while n_ < len(self.ytr):
                w,n_ = self.train_batch(self.X_tr,self.ytr,w,alpha,epsilon,n,n_)        
        train_err = self.fMSE(self.X_tr,self.ytr,w,alpha)
        test_err = self.fMSE(self.X_te,self.yte,w,alpha)
        return w,train_err , test_err , h_star 


def main():
    #loading the data
    X_tr = np.reshape(np.load("../../HW1/Data/age_regression_Xtr.npy"), (-1, 48*48)) #5000X2304 
    ytr = np.load("../../HW1/Data/age_regression_ytr.npy") #5000 X 1 
    X_te = np.reshape(np.load("../../HW1/Data/age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("../../HW1/Data/age_regression_yte.npy")
    # learning_rates = []
    # epochs = []
    # regularization = []
    # batch_size = []
    sgd =  SGD(X_tr,ytr,X_te,yte)
    model,train_err,test_err,hyper_parameters= sgd.stochastic_gradient_descent()

    print("Train Error: ",train_err)
    print("Test Error: ",test_err)
    print("Batch Size: ",hyper_parameters[0])
    print("Learning Rate: ",hyper_parameters[1])
    print("Epochs: ",hyper_parameters[2])
    print("Regularization: ",hyper_parameters[3])

if __name__ == "__main__":
    main()
