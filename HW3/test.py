import pickle 
import matplotlib.pyplot as plt 
import numpy as np 
from softmax_regression import SMR
def add_bias(X):
        #adding a bias term for the Train and test labels 
    return np.hstack((X,np.ones((X.shape[0],1))))

def create_labels(y):
    out = np.zeros((y.shape[0],10)) #create array of zeros of size of training labels and classes
    for i,val in enumerate(y):
        out[i][val]=1 #set a corresponding class index to 1 
    return out 
def softmax(z):
    exp = np.exp(z) 
    return exp /np.sum(exp,axis=1,keepdims=1)
def predict(X,y,w):
    z = X.T@w 
    # print("Z" , z)
    y_tilde = softmax(z)
    return y_tilde

loaded_model = pickle.load(open("model.sav", 'rb'))

X_te = np.reshape(np.load("Data/fashion_mnist_test_images.npy"), (-1, 28*28))
yte = np.load("Data/fashion_mnist_test_labels.npy")

X_te = add_bias(X_te).T /255 #adding bias to testing labels , and transposing to reflect the theory 
yte = create_labels(yte)

y_tilde= predict(X_te,yte,loaded_model)

# y_tilde[np.argmax(y_tilde,axis=1)] = 1
# y_tilde[y_tilde!=1] = 0
y_tilde = (y_tilde == y_tilde.max(axis=1)[:,None]).astype(float)

print("Test", yte)
print("predict" ,y_tilde)
print("Acc", np.sum((yte == y_tilde).all(1)*100/yte.shape[0]))
# print(np.argmax(y_tilde,axis=1))

out  = np.zeros((28,1))
for i in range(10):
    # i = 5
    w = loaded_model[:-1,i]
    # w1 *= (255.0/w1.max())
    w = w.reshape((28,28))
    out = np.hstack((out,w))

plt.imshow(out,cmap="gray")
plt.show()
i+=1




