
# coding: utf-8

# ## Cross entropy
# 
# -  A simple way to understand cross entropy is to consider having a set of events and set of probabilities, and posing the question - how likely is it that these events happened based on these probabilities? If the answer is very likely, then the cross entropy is *small* , otherwise it is large.
# -  To calculate cross entropy, we take the sum of the negatives of the logarithms of the probabilities.
# -  Using the sum of logarithms allows us to avoid taking the product of probabilities which could result in a very small quanitity.
# -  Since the log of a number between 0 and 1 (since we are using only these kinds of values) is always negative, we in turn have take the negative of the log of the probability.
# -  A low cross entropy indicates a good model - this is because a good model gives a high probability, but we are actually taking the negative log of that high number which results in a small number!
# -  This also applies to the individual points, correctly classified points will have smaller values or errors.
# -  The cross entropy can be extended for a multi class problem as well.

# In[ ]:


import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

