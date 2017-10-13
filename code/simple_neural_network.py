
# coding: utf-8

# ## Simple neural network & activation functions
# 
# -  In the perceptron example, the activation function was a simple step function which only gave an output of either a 0 or 1.
# -  Another type of activation function is the *sigmoid function* which is implemented below. 
# -  The output of the sigmoid function is a value between 0 and 1 so it seems like a probability and one can make a decision of whether the prediction is a 0 or 1 (another terminology is negative or positive class) based on how close the value is to those possible labels. 
# -  Note: the 2 output values for binary classification will always sum to 1.

# In[ ]:


import numpy as np

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1 / (1 + np.exp(-x))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# TODO: Calculate the output
output = sigmoid(np.dot(weights, inputs) + bias)

print('Output:')
print(output)


# -  For multi class classification - predicting labels for more than 2 classes, a *softmax function* can be used. Again, the values will sum up to 1 at the output layer.
# -  The output layer this time takes an n-dimensional array of values, applies the softmax function, and returns an equal-sized array of the same shape.

# In[ ]:


def softmax(z):
    """Compute softmax values for each sets of scores in z."""
    return np.exp(z) / np.sum(np.exp(z), axis=0)

logits = [3.0, 1.0, 0.2]
print(softmax(logits))

