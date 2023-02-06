import numpy as np
import matplotlib.pyplot as plt 
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def CE_loss(y_hat,y):
    n = y.shape[0]
    return -(1/n)*np.sum(np.sum(y*np.log(y_hat),axis=0))

weights = []
biases = [] 

training_loss = []

itr = 1000
alpha = 1
w =0
b= -100
train_X = np.array([[97.7],
                    [98.1],
                    [98.4],
                    [97.3],
                    [97.1]]
                    )
train_y = np.array([[1],
                    [1],
                    [1],
                    [1],
                    [1]])

test_X = np.array([[99.3],
                   [99.7],
                   [96.4],
                    [98.1],
                    [97.5]])

test_Y = np.array([[0],
                   [0],
                    [0],
                    [1],
                    [1]])

for i in range(itr):
    
    y_hat = sigmoid(train_X*w+b)
    training_loss.append(CE_loss(y_hat,train_y))

    grad_w = np.mean(train_X*(y_hat-train_y))
    grad_b = np.mean(y_hat-train_y)

    w = w- alpha*grad_w
    b = b-alpha*grad_b 
    print(w,b)

    weights.append(w)
    biases.append(b)
# print(w,b)
# print(sigmoid(train_X*w+b))
# plt.plot(weights,label="Weight")
# plt.plot(biases,label="bias")
# plt.legend()
# plt.show()

# plt.plot(training_loss,label="Train Loss")
# plt.legend()
# plt.show()

# Testing 

# testing_loss = 
test_y_hat = sigmoid(test_X*w+b)
print(test_y_hat*np.log(test_Y))
print(test_y_hat)

print("Testing_loss",CE_loss(test_y_hat,test_Y))









