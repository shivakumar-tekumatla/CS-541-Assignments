import numpy as np 


class SGD:
    def __init__(self,X_tr,ytr,X_te,yte,n=[100,200,300],epsilon=[0.001,0.1,1],epochs=[2,3,4],alpha=[2,3,4]) -> None:

        self.X_tr = self.add_bias(X_tr).T #adding bias to training labels , and transposing to reflect the theory
        self.ytr = ytr 
        self.X_te = self.add_bias(X_te).T #adding bias to testing labels , and transposing to reflect the theory
        self.yte = yte 
        self.H = np.array(np.meshgrid(n,epsilon,epochs,alpha)).T.reshape(-1,4) #creating combination of all the hyper parameters 

        pass
    def gradient(self,X,w,y,alpha):
        # here w is embedded with bias , and that is the last weight 
        # L2 Regularized gradient 
        # No regularization for the bias term 
        alpha = np.full(w.shape[0],alpha) 
        alpha[-1] = 0  # no regularization for bias term 
        n = y.shape[0] # number of examples 
        # print("alpha",alpha)
        # print("w",w.T)
        # print("alpha w",alpha@w)
        return (1/n)*X@(X.T@w-y)+alpha@w #.T[0]

    def update_weights(self,w,gradient,epsilon):
        print("we",gradient)
        print(w)
        return w-epsilon*gradient

    def add_bias(self,X):
        #adding a bias term for the Train and test labels 
        return np.hstack((X,np.ones((X.shape[0],1))))

    def fMSE(self,X,y,w,alpha):
        # Mean square error
        y = np.expand_dims(y, axis=1)
        # print(alpha*w.T*w)
        # print(X.shape)
        
        # print(y.shpae)
        return (np.sum((X.T@w-y)**2)+alpha*w.T*w)/(2*X.shape[0]) 

    def train(self,X,y,w,alpha,epsilon):
        grad = self.gradient(X,w,y,alpha)
        
        return self.update_weights(w,grad,epsilon)
    def test(self,X,y,w,alpha):
        
        # alpha = np.full(w.shape,alpha) 
        # alpha[-1] = 0  # no regularization for bias term 
        return self.fMSE(X,y,w,alpha)

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

    def stochastic_gradient_descent(self):
        X_tr,ytr,X_va,yva = self.split_train_validation(0.8) # Split 80% 
        print(self.H)
        h_star =self.H[np.random.choice(len(self.H))] # Initially taking a random hyper parameter set as the best one 
        acc = 0 #initial accuracy 
         

        for h in self.H: # for each hyper parameter set 
            w = np.random.uniform( size=(self.X_tr.shape[0],1))#initialize the weights with bias term 
            accuracies = []
            print(w)
            input()
            n,epsilon,epochs ,alpha =  list(map(int,h)) 
            print(n)
            for epoch in range(int(epochs)):
                n_ = 0 
                while n_ < len(ytr):
                    print(ytr[n_:n+n_])
                    # for each batch , train and update the weights 
                    w = self.train(X_tr[:,n_:n+n_],ytr[n_:n+n_],w,alpha,epsilon)
                    print(w)
                    input()
                    n_ += n
            new_acc = self.test(X_va,yva,w,alpha) #testing on the validation dataset 
            accuracies.append(new_acc)
            print(new_acc)
            if  new_acc > acc:
                acc =  new_acc 
                h_star = h  # this is the best h_star so far 
        # Now we found the best hyper parameters 
        # Train in the whole training dataset 
        n,epsilon,epochs ,alpha =  h_star 
        w = np.random.uniform( size=(self.X_tr.shape[0],1))#initialize the weights with bias term 
        for epoch in range(epochs):
            n_ = 0 
            while n_ < len(self.ytr):
                w = self.train(self.X_tr[:,n_:n+n_],self.ytr[n_:n+n_],w,alpha,epsilon)
                n_ += n 

        # finally test
        print(self.test(self.X_te,self.yte,w))

def main():

    #loading the data
    X_tr = np.reshape(np.load("../../HW1/Data/age_regression_Xtr.npy"), (-1, 48*48)) #5000X2304 
    
    ytr = np.load("../../HW1/Data/age_regression_ytr.npy") #5000 X 1 
    X_te = np.reshape(np.load("../../HW1/Data/age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("../../HW1/Data/age_regression_yte.npy")

    sgd =  SGD(X_tr,ytr,X_te,yte)

    # print(sgd.split_train_validation(0.8)[0].shape)
    print(sgd.stochastic_gradient_descent())


if __name__ == "__main__":
    main()