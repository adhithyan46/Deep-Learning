import numpy as np

def relu(x):
    return np.maximum(0,x)
def relu_deriv(x): #derivative for backpropagation
    return np.where(x > 0,1,0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): #derivative for backpropagation
    sig = sigmoid(x)
    return sig * (1 - sig)

#forward propagation

def forward_prop(X,weights,biases):
     
     # input layer to hidden layer
     
     z1 = np.dot(X,weights['w1']) + biases['b1']
     a1 = relu(z1)
     
     #hidden layer to output layer
     
     z2 = np.dot(a1,weights['w2']) + biases['b2']
     a2 = sigmoid(z2)
     
     #storing cache for backpropagation
     
     cache ={'z1' : z1, 'a1' : a1, 'z2' : z2, 'a2' : a2}
     
     return a2 , cache
 
# backpropagation

def back_prop(X,Y,cache,weights,biases,LR = 0.01):
    
    #retreving cache 
    a1,a2 = cache['a1'],cache['a2']
    z1,z2 = cache['z1'],cache['z2']
    
    #calculating the gradient of loss with respect to a2 (output layer)
    
    da2 = -(Y - a2)  #derivative of MSE with respect to a2
    
    #output layer to hidden layer (sigmoid derivative)
    
    dz2 = da2 * sigmoid_deriv(z2)
    dw2 = np.dot(a1.reshape(-1,1),dz2.reshape(1,-1))
    db2 = dz2
    
    #hidden layer to input layer
    
    da1 = np.dot(weights['w2'],dz2)
    dz1 = da1 * relu_deriv(z1)
    dw1 = np.dot(X.reshape(-1,1),dz1.reshape(1,-1))
    db1 = dz1
    
    #updating weight and bias using gradient descent
    
    weights['w1'] -= LR * dw1
    weights['w2'] -= LR * dw2
    biases['b1'] -= LR * db1
    biases['b2'] -= LR * db2
    
#defining weights and biases 
weights ={
    'w1' :np.array([[0.2, -0.5], [0.3, 0.8]]),
    'w2' :np.array([[0.7],[-0.2]])
}

biases ={
    'b1' : np.array([0.1, -0.3]),
    "b2": np.array([0.05])
}

X =np.array([0.5,-1.2]) #input
Y =np.array([1]) #Target output

output,cache = forward_prop(X,weights,biases)
print("forward propagation output :", output)

back_prop(X,Y,cache,weights,biases,LR=0.01) #performing backpropagation

output,_ = forward_prop(X,weights,biases)
print('output after backpropagation :' ,output)




















