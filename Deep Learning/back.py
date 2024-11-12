import numpy as np 

def relu(x):
    return np.maximum(0,x)
def relu_deriv(x):
    return np.where(x >0 ,1,0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    sig = sigmoid(x)
    return sig * ( 1 - sig)

def forward(x,weight,bias):
    z1 = np.dot(x,weight['w1']) + bias['b1']
    a1 = relu(z1)
    
    z2 = np.dot(a1,weight['w2']) + bias['b2']
    a2 = sigmoid(z2)
    
    hist = {'z1' : z1, 'a1' : a1, 'z2' : z2, 'a2' : a2}
    
    return a2 , hist

def back(x,y,hist,weight,bias,lr=0.01):
    
    a1 = hist['a1']
    a2 = hist['a2']
    z1,z2 = hist['z1'],hist['z2']
    
    da2 = -(y-a2)
    
    dz2 = da2 * sigmoid_deriv(z2)
    dw2 = np.dot(a1.reshape(-1, 1),dz2.reshape(1, -1))
    db2 = dz2
    
    da1 = np.dot(weight['w2'],dz2)
    dz1 = da1 * relu_deriv(z1)
    dw1 = np.dot(x.reshape(-1, 1),dz1.reshape(1, -1))
    db1 = dz1
    
    weight['w1'] -= lr * dw1
    weight['w2'] -= lr * dw2
    bias['b1'] -= lr * db1
    bias['b2'] -= lr *db2
    
    
weight ={
    'w1' : np.array([[0.2,-0.5],[0.3,0.8]]),
    'w2' : np.array([[0.7],[-0.2]])
}

bias ={
    'b1' : np.array([0.1,-0.3]),
    'b2' : np.array([0.05])
}

x = np.array([0.5,-1.2])
y = np.array([1])

output,hist = forward(x,weight,bias)
print('output before back propagation :',output)

back(x,y,hist,weight,bias,lr = 0.01)

output,_ = forward(x,weight,bias)
print('output after back propagation :',output)