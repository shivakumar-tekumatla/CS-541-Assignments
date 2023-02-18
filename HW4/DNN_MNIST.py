import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt 
import math 

class HyperParameter:
    def __init__(self,h) -> None:
        
        pass
class DNN:
    def __init__(self,X_tr,ytr,X_te,yte,c=10,hidden_layers=[6],hidden_units=[128],epsilon=[0.09],n=[128],epochs=[10,20],alpha=[0.0025],validation_split =0.8,tune = True) -> None:
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
        self.tune = tune 
        self.h_star = self.tuning()
        # self.forward_propagation(self.H[0])
        # h_star = self.tuning()
        # self.Ws,self.bs = self.train(X_tr,ytr,h_star,regularize=True)
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
        return Ws,bs

    def pack(self,Ws,bs):
        # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
        # pack the weights and biases into one vector 
        return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    
    def unpack(self,weights,no_hidden_layers,no_hidden_units):
        # unpack the weights and biases from vector form to original form 
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

    def ReLU_prime(self, z): 
        # derivative of relu activation function 
        return 1 * (z > 0)
    
    def softmax(self,z):
        # softmax activation 
        exp = np.exp(z) 
        return exp /np.sum(exp,axis=1,keepdims=1)

    def fCE(self,X,y,Ws,bs,alpha,regularize):
        y_tilde,zs,hs= self.forward_propagation(X,y,Ws,bs)
        # print(y_tilde)
        n = y.shape[0] 
        # Cross entropy loss 
        unreg_ce = -(1/n)*np.sum(y*np.log(y_tilde))
        if regularize:
            # regularizing only w.r.t weights 
            w = np.hstack([ W.flatten() for W in Ws ])
            return unreg_ce -(1/n)*(0.5*alpha*np.sum(np.square(w))),y_tilde 
        return unreg_ce,y_tilde 

    def grad_fCE(self,X,y,Ws,bs,alpha,regularize):
        y_tilde = self.forward_propagation(X,Ws,bs)
        n = y.shape[0] 
        if regularize:

            pass
        
        return (1/n)*X@(y_tilde-y)

    def forward_propagation(self,X,y,Ws,bs):
        # ReLU activation is used for every layer except the last layer 
        # For last layer it is soft max activation 
        zs = []
        hs = [] 
        h = X.T 
        for i,(w,b) in enumerate(zip(Ws,bs)):
            z = h@w+b
            zs.append(z)
            if i != len(Ws)-1:
                h = self.ReLU(z) 
                hs.append(h)
            else:
                # for the last layer use softmax activation 
                h = self.softmax(z)
        return h,zs,hs # this is same as y_hat for this propagation 
    
    def backward_propagation(self,X,y,Ws,bs,alpha,no_hidden_layers,regularize = True):
        y_tilde,zs,hs = self.forward_propagation(X,y,Ws,bs)
        # this is where we update the weights based on the gradient 
        dJdWs = len(Ws)*[[]]#np.zeros_like(Ws)#[]  # Gradients w.r.t. weights
        dJdbs = len(bs)*[[]]#np.zeros_like(bs)#[]  # Gradients w.r.t. biases
        g = y_tilde-y  # 200X10 
        n = g.shape[0]
        # print("g",g.shape)
        for k in range(no_hidden_layers, -1, -1):
            if regularize:
                reg = 2*alpha*Ws[k] 
            else:
                reg = 0
            dJdb = np.mean(g,axis=0) 
            dJdbs[k] = dJdb
            if k==0:
                dJdW = X@ g + reg
            else:
                dJdW = hs[k-1].T @ g + reg 

            dJdWs[k] = dJdW/n 
            if k !=0:
                g = g@Ws[k].T  #200X30 
                g =  g*self.ReLU_prime(zs[k-1]) # 

        return dJdWs,dJdbs
    def update_weights(self,X,y,Ws,bs,alpha,no_hidden_layers,epsilon,regularize = True):
        dJdWs,dJdbs = self.backward_propagation(X,y,Ws,bs,alpha,no_hidden_layers,regularize)
        for i in range(len(Ws)):
            Ws[i] = Ws[i] - epsilon*dJdWs[i]
            bs[i] = bs[i] - epsilon*dJdbs[i]
        return Ws,bs

    def train(self,X_tr,ytr,h,regularize=True):
        no_hidden_layers,no_hidden_units,n,epsilon,epochs,alpha = h
        print(f'Using hyper parameters Batch Size = {n}, Epsilon = {epsilon}, epochs = {epochs}, alpha = {alpha}, hidden layers= {no_hidden_layers},hodden_units = {no_hidden_units}')
        no_hidden_layers = int(no_hidden_layers)
        no_hidden_units = no_hidden_layers*[int(no_hidden_units)]
        Ws,bs = self.initWeightsAndBiases(int(no_hidden_layers),no_hidden_units)
        n = int(n) 
        epochs = int(epochs) 
        for epoch in range(epochs):
            n_ = 0 
            while n_ < len(ytr):
                # Split the batch 
                X = X_tr[:,n_:n+n_]
                y = ytr[n_:n+n_]
                Ws,bs = self.update_weights(X,y,Ws,bs,alpha,no_hidden_layers,epsilon,regularize)
                n_+=n
        fCE = self.fCE(X,y,Ws,bs,alpha,regularize)[0] #get only train error 
        return Ws,bs,fCE

    def test(self,Ws,bs,h,regularize=True):
        alpha = h[-1]
        # print(self.fCE(self.X_te,self.yte,Ws,bs,alpha,regularize))
        ce_loss,y_tilde = self.fCE(self.X_te,self.yte,Ws,bs,alpha,regularize)
        # ce_loss = 0
        # y_tilde,_,_ = self.forward_propagation(self.X_te,self.yte,Ws,bs)
        y_tilde = (y_tilde == y_tilde.max(axis=1)[:,None]).astype(float)
        accuracy = np.sum((self.yte == y_tilde).all(1)*100/self.yte.shape[0])

        # print(accuracy)

        return ce_loss,accuracy


    def tuning(self):
        # this function finds out the best hyper parameter set 
        X_tr,ytr,X_va,yva = self.split_train_validation(self.validation_split) # Split train to train and validation  
        # print(self.H)
        h_star =self.H[np.random.choice(len(self.H))] # Initially taking a random hyper parameter set as the best one 
        if self.tune:
            err = np.inf #initial error
            for h in self.H: # for each hyper parameter set 
                # print()
                Ws,bs,train_fCE = self.train(X_tr,ytr,h,regularize=False) # we do not have to regularize while tuning 
                # print("CE Error ",fCE)
                # now check the error on validation data set 
                alpha = h[-1]
                curr_err,_ = self.fCE(X_va,yva,Ws,bs,alpha,False)
                print("Training Error: ",train_fCE)
                print("Validation Error: ", curr_err) #, h)
                if curr_err <err:
                    h_star = h  #storing as the best hyper parameter set 
                    err = curr_err
        return h_star # we got the best hyper parameters . Now train using these parameters on the whole data 
    def show_weights(self,Ws):
        def multiples(i):
            for k in range(math.ceil(math.sqrt(i)), 0, -1): 
                # finding the factors of i with least sum. 
                # this helps us in reshaping the weights properly if there is no integer sqrt of a number 
                if i % k == 0:
                    m1, m2 =k,int(i / k) 
                    return m1,m2 

        # W = W.T 
        for layer,W in enumerate(Ws):
            i,j = W.shape 
            m1,m2 = multiples(i) 
            n1,n2 = multiples(j)
            # n = int(j ** 0.5)
            plt.title(f"Weights at Layer:{layer+1}")
            plt.imshow(np.vstack([np.hstack([ np.pad(np.reshape(W[:,idx1*n1 + idx2],[ m1,m2]), 2, mode='constant') for idx2 in range(n2) ]) for idx1 in range(n1)]), cmap='gray')
            plt.show()

def main():
    X_tr = np.reshape(np.load("../HW3/Data/fashion_mnist_train_images.npy"), (-1, 28*28)) 
    ytr = np.load("../HW3/Data/fashion_mnist_train_labels.npy") 
    X_te = np.reshape(np.load("../HW3/Data/fashion_mnist_test_images.npy"), (-1, 28*28))
    yte = np.load("../HW3/Data/fashion_mnist_test_labels.npy")

    dnn = DNN(X_tr,ytr,X_te,yte,tune=False)
    # h = dnn.H[0]#dnn.tune()
    print("Found the best hyper parameter set " , dnn.h_star)
    Ws,bs,train_error = dnn.train(dnn.X_tr,dnn.ytr,dnn.h_star)

    print("Training Error is " ,train_error)

    test_error,test_Acc = dnn.test(Ws,bs,dnn.h_star)
    print("Test Error is ",test_error)
    print("Test accuracy",test_Acc)
    # for i in range(4):
    dnn.show_weights(Ws)#[i])#,128)


if __name__=="__main__":
    main()