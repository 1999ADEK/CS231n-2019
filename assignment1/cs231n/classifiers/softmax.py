from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    for i in range(num_train):
        scores = X[i].dot(W) # (C,)
        e_scores = np.exp(scores) # (C,)
        e_sum = e_scores.sum() # (1,)
        correct_e_score = e_scores[y[i]] # (1,)
            
        prob = correct_e_score / e_sum # (1,)
        loss += -np.log(prob) # (1,)
        
        grad_prob = -1 / prob # (1,)
        grad_soft = -correct_e_score * np.ones(scores.shape)
        grad_soft[y[i]] += e_sum
        grad_soft = grad_prob * (grad_soft) / (e_sum**2)  * e_scores # (C,)
        dW += X[i].reshape((-1, 1)) @ grad_soft.reshape((1, -1))
        
        
    
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    scores = X @ W # (N, C)
    scores -= scores.max() # for numerical stability
    e_scores = np.exp(scores) # (N, C)
    correct_e_scores = e_scores[range(num_train), y] # (N,)
    e_sum = e_scores.sum(axis=1) # (N,)
    prob = correct_e_scores / e_sum # (N,)
    L = -np.log(prob) # (N,)
    loss = L.sum() / num_train
    
    grad_soft = e_scores / e_sum.reshape(-1, 1)
    grad_soft[range(num_train), y] -= 1
    dW = X.T @ grad_soft / num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
