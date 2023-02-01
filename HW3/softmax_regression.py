import numpy as np 
import pickle 
class SMR:
    def __init__(self,X_tr,ytr,X_te,yte,c =10,n=[1000,2000,4000],epsilon=[0.01,0.1,0.1,0.2],epochs=[2,4,6],alpha=[1,5,10,20],validation_split =0.8) -> None:
        self.c = c #classification 
        self.X_tr = self.add_bias(X_tr).T /255 #adding bias to training labels , and transposing to reflect the theory. Normaling pixels 
        self.ytr = self.create_labels(ytr)
        self.X_te = self.add_bias(X_te).T /255 #adding bias to testing labels , and transposing to reflect the theory . Normalizing pixels 
        self.yte = self.create_labels(yte) 
        self.H = np.array(np.meshgrid(n,epsilon,epochs,alpha)).T.reshape(-1,4) #creating combination of all the hyper parameters 
        self.validation_split = validation_split

        # print(X_tr.shape)
        
        pass
    def add_bias(self,X):
        #adding a bias term for the Train and test labels 
        return np.hstack((X,np.ones((X.shape[0],1))))

    def create_labels(self,y):
        out = np.zeros((y.shape[0],self.c)) #create array of zeros of size of training labels and classes
        for i,val in enumerate(y):
            out[i][val]=1 #set a corresponding class index to 1 
        return out 
    def split_train_validation(self,split):
        # Randomize the train data 
        allIdxs = np.arange(self.X_tr.shape[1]) 
        Idxs = np.random.permutation(allIdxs) #random indices for the train data 
        # select the 1st split  of the indices for thr train data and rest for validation 
        train_part = Idxs[:int(len(Idxs)*split)]
        validation_part = Idxs[int(len(Idxs)*split):]
        X_tr = self.X_tr[:,train_part]
        ytr = self.ytr[train_part]
        X_va = self.X_tr[:,validation_part]
        yva = self.ytr[validation_part]
        return X_tr,ytr,X_va,yva  

    def softmax(self,z):
        exp = np.exp(z) 
        return exp /np.sum(exp)
    def predict(self,X,y,w):
        z = X.T@w 
        # print("Z" , z)
        y_tilde = self.softmax(z)
        return y_tilde
    def gradient(self,X,y,w,alpha):
        alpha = np.full(w.shape,alpha)  # generating alpha same as weights shape 
        alpha[-1,:] = 0 #set all the bias reg to zero 
        y_tilde = self.predict(X,y,w)
        n = y.shape[0]
        return (1/n)*X@(y_tilde-y) + alpha*w 
    def CE_loss(self,X,y,w,alpha):
        # log = np.log(y_tilde)
        w_temp = w[:-1,:] #to not reg the bias 
        reg = np.sum(w_temp.T@w_temp,axis=0)
        n = y.shape[0]
        y_tilde = self.predict(X,y,w)
        return -(1/n)*np.sum(np.sum(y*np.log(y_tilde),axis=0) + 0.5*alpha*reg)

    def train_batch(self,X,y,w,alpha,epsilon,n,n_):
        # Split the batch 
        X = X[:,n_:n+n_]
        y = y[n_:n+n_]
        # Compute gradient on this batch 
        gradient = self.gradient(X,y,w,alpha)
        # print(gradient)
        # input()
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
            w = np.zeros((self.X_tr.shape[0],self.c))#np.random.uniform( size=(self.X_tr.shape[0],self.c))#initialize the weights with bias term  #
            n = int(n) 
            epochs = int(epochs) 
            for epoch in range(epochs):
                n_ = 0 
                while n_ < len(ytr):
                    w,n_ = self.train_batch(X_tr,ytr,w,alpha,epsilon,n,n_) #Train on the batch
                    # print(w)
        #     # test on validation data set 
            curr_err = self.CE_loss(X_va,yva,w,alpha)
            print("Error: ", curr_err) #, h)
            if curr_err <err:
                h_star = h  #storing as the best hyper parameter set 
                err = curr_err

        # Now we found the best hyper parameter set from the given data 
        # Again train on the whole data 
        n,epsilon,epochs ,alpha =h_star #best hyper parameters 
        w = np.random.uniform( size=(self.X_tr.shape[0],self.c)) #np.zeros((self.X_tr.shape[0],self.c))#np.random.uniform( size=(self.X_tr.shape[0],1))#initialize the weights with bias term 
        n = int(n) 
        epochs = int(epochs)
        print("The actual training starts....!")      
        for epoch in range(int(epochs)):
            # print(f"Epoch number {epoch}")
            n_ = 0 
            while n_ < len(self.ytr):
                w,n_ = self.train_batch(self.X_tr,self.ytr,w,alpha,epsilon,n,n_)        
        train_err = self.CE_loss(self.X_tr,self.ytr,w,alpha)
        test_err = self.CE_loss(self.X_te,self.yte,w,alpha)
        return w,train_err , test_err , h_star

def main():
    X_tr = np.reshape(np.load("Data/fashion_mnist_train_images.npy"), (-1, 28*28)) 
    ytr = np.load("Data/fashion_mnist_train_labels.npy") 
    X_te = np.reshape(np.load("Data/fashion_mnist_test_images.npy"), (-1, 28*28))
    yte = np.load("Data/fashion_mnist_test_labels.npy")

    smr = SMR(X_tr,ytr,X_te,yte,c=10) 

    model,train_err,test_err,hyper_parameters= smr.stochastic_gradient_descent()
    #store the model 

    pickle.dump(model, open("model.sav", 'wb'))

    print("Train Error: ",train_err)
    print("Test Error: ",test_err)
    print("Batch Size: ",hyper_parameters[0])
    print("Learning Rate: ",hyper_parameters[1])
    print("Epochs: ",hyper_parameters[2])
    print("Regularization: ",hyper_parameters[3])

if __name__=="__main__":
    main()


