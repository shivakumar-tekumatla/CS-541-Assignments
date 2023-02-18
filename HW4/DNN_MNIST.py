import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt 
import math 
import pickle 

class HyperParameter:
    def __init__(self,h) -> None:
        self.no_hidden_layers,self.no_hidden_units,self.n,self.epsilon,self.epochs,self.alpha,self.decay_factor,self.decay_frequency = h
        self.no_hidden_layers = int(self.no_hidden_layers)
        self.no_hidden_units = self.no_hidden_layers*[int(self.no_hidden_units)]
        self.n = int(self.n)
        self.epochs = int(self.epochs) 
        pass
class DNN:
    def __init__(self,X_tr,ytr,X_te,yte,H,c=10,validation_split =0.8,tune = True) -> None:
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
        self.H = H #hyper parameter set 
        self.validation_split = validation_split
        self.tune = tune  # should we tune with the given hyper parameter set or can a random hyperparameter set be chosen ? 
        self.h_star = self.tuning()
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

    def initWeightsAndBiases(self,h):
        # no_hidden_layers = h.no_hidden_layers
        # no_hidden_units  = h.no_hidden_units
        # no_hidden_layers - Number of hidden layers 
        # no_inputs - Number of inputs
        # no_outputs - Number of outputs 
        # no_hidden_units - Number of neurons in each hidden layer - If there are 3 layers , each of them can have different neurons , and no_units[0] gives the hidden neurons in the first hidden layer
        # It is a good practice to have weights initiated to random samples of mean zero and std of 1/sqrt(inputs)
        Ws =[] # Weights
        bs = [] # Biases 
        np.random.seed(0)
        if h.no_hidden_layers == 0:
            # if no hidden layers 
            W = 2*(np.random.random(size=(self.no_inputs,self.no_outputs))/self.no_inputs**0.5) - 1./self.no_inputs**0.5
            b = 0.01 * np.ones(self.no_outputs)
            Ws.append(W)
            bs.append(b)
        else:
            # These are the W and b for the first layer
            W = 2*(np.random.random(size=(self.no_inputs,h.no_hidden_units[0]))/self.no_inputs**0.5) - 1./self.no_inputs**0.5
            Ws.append(W)
            b = 0.01 * np.ones(h.no_hidden_units[0])
            bs.append(b)
            # W and b for all the hidden layers 
            for i in range(h.no_hidden_layers - 1):
                W = 2*(np.random.random(size=(h.no_hidden_units[i], h.no_hidden_units[i+1]))/h.no_hidden_units[i]**0.5) - 1./h.no_hidden_units[i]**0.5
                Ws.append(W)
                b = 0.01 * np.ones(h.no_hidden_units[i+1])
                bs.append(b)
            # output layer 

            W = 2*(np.random.random(size=(h.no_hidden_units[-1],self.no_outputs))/h.no_hidden_units[-1]**0.5) - 1./h.no_hidden_units[-1]**0.5
            Ws.append(W)
            b = 0.01 * np.ones(self.no_outputs)
            bs.append(b)
        return Ws,bs

    def pack(self,Ws,bs):
        # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
        # pack the weights and biases into one vector 
        return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    
    def unpack(self,weights,h):#no_hidden_layers,no_hidden_units):
        # unpack the weights and biases from vector form to original form 
        # Unpack arguments
        no_hidden_layers = h.no_hidden_layers
        no_hidden_units = h.no_hidden_units
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

    def fCE(self,X,y,Ws,bs,h,regularize):
        # cross entropy loss 
        y_tilde,zs,hs= self.forward_propagation(X,y,Ws,bs)
        # print(y_tilde)
        n = y.shape[0] 
        # Cross entropy loss 
        unreg_ce = -(1/n)*np.sum(y*np.log(y_tilde))
        if regularize:
            # regularizing only w.r.t weights 
            w = np.hstack([ W.flatten() for W in Ws ])
            return unreg_ce +(1/n)*(0.5*h.alpha*np.sum(np.square(w))),y_tilde 
        return unreg_ce,y_tilde 

    def forward_propagation(self,X,y,Ws,bs):
        # ReLU activation is used for every layer except the last layer 
        # For last layer it is soft max activation 
        pre_activations = []
        activations = [] 
        act = X.T 
        for i,(w,b) in enumerate(zip(Ws,bs)):
            z = act@w+b
            pre_activations.append(z)
            if i != len(Ws)-1:
                act = self.ReLU(z) 
                activations.append(act)
            else:
                # for the last layer use softmax activation 
                act = self.softmax(z)
        return act,pre_activations,activations # this is same as y_hat for this propagation 
    
    def backward_propagation(self,X,y,Ws,bs,h,regularize = True):
        y_tilde,pre_activations,activations = self.forward_propagation(X,y,Ws,bs)
        # this is where we update the weights based on the gradient 
        dJdWs = len(Ws)*[[]]#np.zeros_like(Ws)#[]  # Gradients w.r.t. weights
        dJdbs = len(bs)*[[]]#np.zeros_like(bs)#[]  # Gradients w.r.t. biases
        g = y_tilde-y  # 200X10 
        n = g.shape[0]
        for k in range(h.no_hidden_layers, -1, -1):
            if regularize:
                reg = 2*h.alpha*Ws[k] 
            else:
                reg = 0
            dJdb = np.mean(g,axis=0) 
            dJdbs[k] = dJdb
            if k==0:
                dJdW = X@ g + reg
            else:
                dJdW = activations[k-1].T @ g + reg 

            dJdWs[k] = dJdW/n 
            if k !=0:
                g = g@Ws[k].T  #200X30 
                g =  g*self.ReLU_prime(pre_activations[k-1]) # 

        return dJdWs,dJdbs
    def update_weights(self,X,y,Ws,bs,h,regularize = True):
        dJdWs,dJdbs = self.backward_propagation(X,y,Ws,bs,h,regularize)
        for i in range(len(Ws)):
            Ws[i] = Ws[i] - h.epsilon*dJdWs[i]
            bs[i] = bs[i] - h.epsilon*dJdbs[i]
        return Ws,bs

    def train(self,X_tr,ytr,h,regularize=True):
        print(f'Using hyper parameters Batch Size = {h.n}, Epsilon = {h.epsilon}, epochs = {h.epochs}, alpha = {h.alpha}, hidden layers= {h.no_hidden_layers},hidden_units = {h.no_hidden_units}')
        Ws,bs = self.initWeightsAndBiases(h)
        for epoch in range(h.epochs):
            n_ = 0 
            while n_ < len(ytr):
                # Split the batch 
                X = X_tr[:,n_:h.n+n_]
                y = ytr[n_:h.n+n_]
                Ws,bs = self.update_weights(X,y,Ws,bs,h,regularize)
                n_+=h.n
            print(f"Epoch {epoch+1} , and loss {self.fCE(X,y,Ws,bs,h,regularize)[0]}")

        fCE = self.fCE(X,y,Ws,bs,h,regularize)[0] #get only train error 
        return Ws,bs,fCE

    def test(self,Ws,bs,h,regularize=True):
        ce_loss,y_tilde = self.fCE(self.X_te,self.yte,Ws,bs,h,regularize)
        y_tilde = (y_tilde == y_tilde.max(axis=1)[:,None]).astype(float)
        accuracy = np.sum((self.yte == y_tilde).all(1)*100/self.yte.shape[0])
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
                curr_err,_ = self.fCE(X_va,yva,Ws,bs,h,False)
                print("Training Error: ",train_fCE)
                print("Validation Error: ", curr_err) #, h)
                if curr_err <err:
                    h_star = h  #storing as the best hyper parameter set 
                    err = curr_err
        return h_star # we got the best hyper parameters . Now train using these parameters on the whole data 
    def show_weights(self,Ws):
        def multiples(i):
            # finding the factors of i with least sum. 
            # this helps us in reshaping the weights properly if there is no integer sqrt of a number 
            for k in range(math.ceil(math.sqrt(i)), 0, -1): 
                if i % k == 0:
                    m1, m2 =k,int(i / k) 
                    return m1,m2 

        for layer,W in enumerate(Ws):
            i,j = W.shape 
            print(i,j)
            m1,m2 = multiples(i) 
            n1,n2 = multiples(j)
            # n = int(j ** 0.5)
            # n1 = n2 = n
            plt.title(f"Weights at Layer:{layer+1}")
            plt.imshow(np.vstack([np.hstack([ np.pad(np.reshape(W[:,idx1*n1 + idx2],[ m1,m2]), 2, mode='constant') for idx2 in range(n2) ]) for idx1 in range(n1)]), cmap='gray')
            plt.show()
    def check_grad(self,X,y,h,Ws,bs):
        weights = self.pack(Ws,bs)
        def function(X,y,weights):
            Ws,bs = self.unpack(weights,h)
            return self.fCE(X,y,Ws,bs,h,False)[0]
        def function_prime(X,y,weights):
            Ws,bs = self.unpack(weights,h)
            dJdWs,dJdbs = self.backward_propagation(X,y,Ws,bs,h,regularize = False)
            return self.pack(dJdWs,dJdbs)
        f = lambda wab:function(X,y,wab)
        f_prime = lambda wab:function_prime(X,y,wab)
        print("Approx f prime   :",scipy.optimize.approx_fprime(weights, lambda weights_: function(X,y,weights_), 1e-6))
        return scipy.optimize.check_grad(f,f_prime,weights)

def main():
    X_tr = np.reshape(np.load("../HW3/Data/fashion_mnist_train_images.npy"), (-1, 28*28)) 
    ytr = np.load("../HW3/Data/fashion_mnist_train_labels.npy") 
    X_te = np.reshape(np.load("../HW3/Data/fashion_mnist_test_images.npy"), (-1, 28*28))
    yte = np.load("../HW3/Data/fashion_mnist_test_labels.npy")
    hidden_layers=[6]
    hidden_units=[254]
    epsilon=[0.09]
    n=[254]
    epochs=[200]
    alpha=[0.0025]
    decay_factor = [1]
    decay_frequency = [1] # times per batch 
    H= list(map(HyperParameter,np.array(np.meshgrid(hidden_layers,hidden_units,n,epsilon,epochs,alpha,decay_factor,decay_frequency)).T.reshape(-1,8))) #creating combination of all the hyper parameters #creating combination of all the hyper parameters 
    dnn = DNN(X_tr,ytr,X_te,yte,H,tune=False)
    print("Found the best hyper parameter set " , dnn.h_star)
    Ws,bs,train_error = dnn.train(dnn.X_tr,dnn.ytr,dnn.h_star)
    pickle.dump((Ws,bs), open("model.sav", 'wb'))
    print("Training Error is " ,train_error)
    test_error,test_Acc = dnn.test(Ws,bs,dnn.h_star)
    print("Test Error is ",test_error)
    print("Test accuracy",test_Acc)
    # Ws,bs = pickle.load(open("model.sav", 'rb'))
    dnn.show_weights(Ws)#[i])#,128)
    # print("Check grad : ",dnn.check_grad(dnn.X_tr[:,0:1],dnn.ytr[0:1,:],dnn.h_star,Ws,bs))


if __name__=="__main__":
    main()