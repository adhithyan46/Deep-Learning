import numpy as np 

#relu function for hidden layers
def relu(x):
    return np.maximum(0,x)
#sigmoid function for output layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#forward propagation function
def forward_prop(X,weights,biases):
    
    #input layer to hidden layer
    
    Z1 = np.dot(X , weights['w1']) + biases['b1']
    A1 = relu(Z1)
    
    #hidden layer to output layer
    
    Z2 = np.dot(A1 , weights['w2']) + biases['b2']
    A2 = sigmoid(Z2)
    
    return A2

weights = {
    'w1' : np.array([[0.2,-0.5],[0.3,0.8]]) , #weights for hidden layer
    'w2' : np.array([[0.7],[-0.2]]) # weights for output layer
}    

biases = {
    'b1' : np.array([0.1,-0.3]), # bias for hidden layer
    'b2' : np.array([0.05]) #bias for output layer
}

#inputs

X = np.array([0.5,-1.2])

output = forward_prop(X,weights,biases)

print('Output : ',output)