#step_function

# thresold = 1.5 
# inputs = [0,0,1,0,1]
# weights = [0.7,0.6,0.5,0.3,0.4]
# sum = 0
# for i in range(len(inputs)):
#     sum+= inputs[i] * weights[i]
# def step_function(sum):
#     return 1 if sum > thresold else 0     #f(x)={1 if x>0, 0 otherwise
# print(step_function(sum))


#sigmoid function

import numpy as np

# thresold = 1.5
# inputs = [0,0,1,0,1]
# sum = 0 
# for i in range(len(inputs)):
#     sum += inputs[i] * weights[i]

# def sigmoid_fun(sum):
#     return 1 / 1 + np.exp(-sum)
# print(sigmoid_fun(sum))

#Tanh function

# thresold = 1.5
# inputs = [0,0,1,0,1]
# weights = [0.7,0.6,0.5,0.3,0.4]
# sum = 0 
# for i in range(len(inputs)):
#     sum += inputs[i] + weights[i]
    
# def tanh_fun(sum):
#     return np.tanh(sum)
# print(tanh_fun(sum))

# ploting graph with tanh function

# import seaborn as sns
# import matplotlib.pyplot as plt
# inputs = np.linspace(-5,5,100)
# tanh = np.tanh(inputs)
# sns.lineplot(x = inputs , y = tanh)
# plt.xlabel('inputs')
# plt.ylabel('tanh')
# plt.title('Tanh function')
# plt.show()

#ReLU function
 
# inputs = [1,0,1,0,1]
# weights =[0.7,0.6,0.5,0.3,0.4]
# sum = 0 
# for i in range(len(inputs)):
#     sum += inputs[i] * weights[i]
# def relu_fun(x):
#     return max(0,x)
# print(relu_fun(sum))


#plotting with relu function

# import matplotlib.pyplot as plt
# import seaborn as sns
# def relu_func(x):
#     return np.maximum(0,x)

# x = np.linspace(-5,5,100)
# y = relu_func(x)
# plt.plot(x ,y,label = 'ReLU' )
# plt.xlabel('Values')
# plt.ylabel('Outputs')
# plt.title('ReLU function')
# plt.show()

#leakyReLU function

# inputs = [0,0,1,0,1]
# weights = [0.7,0.6,0.5,0.3,0.4] 
# sum = 0
# for i in range(len(inputs)):
#     sum += inputs[i] * weights[i]
# def leaky_ReLU_func(x,alpha = 0.01):
#     return x if x > 0 else alpha * x
# print(leaky_ReLU_func(sum))


#softmax function

# def softmax_func(x):
#     ex = np.exp(x - np.max(x))
#     return ex / ex.sum(axis = 0)
# inputs = np.array([1,2,3])
# print(softmax_func(inputs))

#plotting with softmax function 

# import matplotlib.pyplot as plt
# import seaborn as sns
# def softmax_func(x):
#     ex = np.exp(x - max(x))
#     return ex / ex.sum(axis = 0)
# inputs = np.linspace(-5,5,100)
# y = softmax_func(inputs)
# plt.plot( inputs , y )
# plt.xlabel('inputs')
# plt.ylabel('outputs')
# plt.title('softmax function')
# plt.show()