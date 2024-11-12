import numpy as np

w = 0.0 #initial weight
b = 0.0 #initial bias
alpha = 0.01  #learning rate
epochs = 1000  # number of iteretion

x = np.array([1,2,3,4])  #Inputs
y = np.array([2,4,6,8]) #Actual values

#Gradient descent

for epoch in range(epochs):
    
    y_pred = w * x + b  # predicting :Forward propagation
    
    #calculating loss (MSE)
    
    loss = ((y - y_pred) ** 2)
    
    #calculating gradient
    
    dl_dw = -2 * ((y - y_pred) * x) # gradient respect to weight
    dl_db = -2 * ((y - y_pred) * 1) # gradient respect to bias
    
    #updating parameters
    
    w -= alpha * dl_dw # updated weight
    b -= alpha * dl_db # updated bias
    
    
print('Optimized weight :',w)
print('Optimized bias :',b) 
print('Loss :',loss)      








