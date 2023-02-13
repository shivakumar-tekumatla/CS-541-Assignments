import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt 

class DNN:
    def __init__(self,X_tr,ytr,X_te,yte,c=10,hidden_layers=[3,4,5],hidden_units=[30,40,50],epsilon=[0.001,0.005,0.01,0.05,0.1,0.5],n=[200,300],epochs=[200,300],alpha=[0.001,0.01],validation_split =0.8) -> None:
        # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
        # 0.5 (so that the range is [-0.5,+0.5]).
        self.c = c #c classes  
        X_tr = X_tr/255 -0.5 # Normalizing pixels 
        X_te = X_te/255 -0.5  # Normalizing pixels 
        self.no_inputs = X_tr.shape[1] # the number of inputs same as feature size 
        self.no_outputs = c 
        self.X_tr = X_tr.T#self.add_bias(X_tr).T  #adding bias to training labels , and transposing to reflect the theory. 
        self.ytr = self.create_labels(ytr)
        self.X_te = X_te.T#self.add_bias(X_te).T  #adding bias to testing labels , and transposing to reflect the theory 
        self.yte = self.create_labels(yte) 
        self.H = np.array(np.meshgrid(hidden_layers,hidden_units,n,epsilon,epochs,alpha)).T.reshape(-1,6) #creating combination of all the hyper parameters 
        self.validation_split = validation_split
        # self.forward_propagation(self.H[0])
        self.tuning()
        pass
    def add_bias(self,X):
        # adding a bias term for the Train and test labels 
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

    def initWeightsAndBiases(self,no_hidden_layers,no_hidden_units):
        # no_hidden_layers - Number of hidden layers 
        # no_inputs - Number of inputs
        # no_outputs - Number of outputs 
        # no_hidden_units - Number of neurons in each hidden layer - If there are 3 layers , each of them can have different neurons , and no_units[0] gives the hidden neurons in the first hidden layer
        # It is a good practice to have weights initiated to random samples of mean zero and std of 1/sqrt(inputs)
        Ws =[] # Weights
        bs = [] # Biases 
        np.random.seed(0)
        # These are the W and b for the first layer
        W = 2*(np.random.random(size=(self.no_inputs,no_hidden_units[0]))/self.no_inputs**0.5) - 1./self.no_inputs**0.5
        Ws.append(W)
        b = 0.01 * np.ones(no_hidden_units[0])
        bs.append(b)
        # W and b for all the hidden layers 
        for i in range(no_hidden_layers - 1):
            W = 2*(np.random.random(size=(no_hidden_units[i], no_hidden_units[i+1]))/no_hidden_units[i]**0.5) - 1./no_hidden_units[i]**0.5
            Ws.append(W)
            b = 0.01 * np.ones(no_hidden_units[i+1])
            bs.append(b)
        # output layer 

        W = 2*(np.random.random(size=(no_hidden_units[-1],self.no_outputs))/no_hidden_units[-1]**0.5) - 1./no_hidden_units[-1]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(self.no_outputs)
        bs.append(b)
        # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
        weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
        return weights

    def unpack(self,weights,no_hidden_layers,no_hidden_units):
        # Unpack arguments
        Ws = []

        # Weight matrices
        start = 0
        end = self.no_inputs*no_hidden_units[0] #NUM_INPUT*NUM_HIDDEN[0]
        W = weights[start:end]
        Ws.append(W)

        # Unpack the weight matrices as vectors
        for i in range(no_hidden_layers - 1):
            start = end
            end = end + no_hidden_units[i]*no_hidden_units[i+1]
            W = weights[start:end]
            Ws.append(W)

        start = end
        end = end + no_hidden_units[-1]*self.no_outputs
        W = weights[start:end]
        Ws.append(W)

        # Reshape the weight "vectors" into proper matrices
        Ws[0] = Ws[0].reshape(self.no_inputs,no_hidden_units[0])
        for i in range(1, no_hidden_layers ):
            # Convert from vectors into matrices
            Ws[i] = Ws[i].reshape(no_hidden_units[i-1], no_hidden_units[i])
        Ws[-1] = Ws[-1].reshape(no_hidden_units[-1],self.no_outputs)

        # Bias terms
        bs = []
        start = end
        end = end + no_hidden_units[0]
        b = weights[start:end]
        bs.append(b)

        for i in range(no_hidden_layers - 1):
            start = end
            end = end + no_hidden_units[i+1]
            b = weights[start:end]
            bs.append(b)

        start = end
        end = end + self.no_outputs
        b = weights[start:end]
        bs.append(b)

        return Ws, bs
        


    def ReLU(self,z):
        # Relu activation function 
        return z * (z > 0) 
    
    def softmax(self,z):
        # softmax activation 
        exp = np.exp(z) 
        return exp /np.sum(exp,axis=1,keepdims=1)
    # def CE_loss(self,X,y,Ws,bs,alpha,regularize=True):
    #     # Cross entropy loss 
    #     n = y.shape[0]
    #     # y_tilde = self.predict(X,y,w)
    #     y_tilde = self.forward_propagation(X,Ws,bs)
    #     if regularize:
    #         w_temp = w[:-1,:] #to not reg the bias 
    #         reg = np.sum(w_temp.T@w_temp,axis=0)
    #         return -(1/n)*np.sum(np.sum(y*np.log(y_tilde),axis=0)+ 0.5*alpha*reg)
    #     else:
    #         return -(1/n)*np.sum(np.sum(y*np.log(y_tilde),axis=0))

    def forward_propagation(self,X,weights,no_hidden_layers,no_hidden_units):
        # ReLU activation is used for every layer except the last layer 
        # For last layer it is soft max activation 
        Ws,bs = self.unpack(weights,no_hidden_layers,no_hidden_units)
        h = X.T 
        for i,(w,b) in enumerate(zip(Ws,bs)):
            print(h.shape)
            print(w.shape)
            print(b.shape)
            z = h@w+b
            if i != len(Ws)-1:
                h = self.ReLU(z) 
            else:
                # for the last layer use softmax activation 
                h = self.softmax(z)

        return h # this is same as y_hat for this propagation 
    
    def backward_propagation(self):
        # this is where we update the weights based the gradient 
        return 

    def train_batch(self,X,y,weights,no_hidden_layers,no_hidden_units,alpha,epsilon,n,n_,regularize=True):
        # Split the batch 
        X = X[:,n_:n+n_]
        y = y[n_:n+n_]
        # forward propagation 
        y_hat = self.forward_propagation(X,weights,no_hidden_layers,no_hidden_units)
        # backward propagation 

        n_+=n
        return weights, n_

    def tuning(self):
        # this function finds out the best hyper parameter set 
        X_tr,ytr,X_va,yva = self.split_train_validation(self.validation_split) # Split train to train and validation  
        # print(self.H)
        h_star =self.H[np.random.choice(len(self.H))] # Initially taking a random hyper parameter set as the best one 
        err = np.inf #initial error
        for h in self.H: # for each hyper parameter set 
            no_hidden_layers,no_hidden_units,n,epsilon,epochs,alpha = h 
            no_hidden_layers = int(no_hidden_layers)
            no_hidden_units = no_hidden_layers*[int(no_hidden_units)]
            weights = self.initWeightsAndBiases(int(no_hidden_layers),no_hidden_units)
            # print("shape",self.forward_propagation(X_tr,Ws,bs)[0])
            n = int(n) 
            epochs = int(epochs) 
            for epoch in range(epochs):
                n_ = 0 
                while n_ < len(ytr):
                    weights,n_ = self.train_batch(X_tr,ytr,weights,no_hidden_layers,no_hidden_units,alpha,epsilon,n,n_,regularize=False) #Train on the batch
        


         
         



def main():
    X_tr = np.reshape(np.load("../HW3/Data/fashion_mnist_train_images.npy"), (-1, 28*28)) 
    ytr = np.load("../HW3/Data/fashion_mnist_train_labels.npy") 
    X_te = np.reshape(np.load("../HW3/Data/fashion_mnist_test_images.npy"), (-1, 28*28))
    yte = np.load("../HW3/Data/fashion_mnist_test_labels.npy")

    dnn = DNN(X_tr,ytr,X_te,yte)
    # dnn.forward_propagation()


if __name__=="__main__":
    main()