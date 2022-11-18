import numpy as np
from numpy import linalg as la
import time as t

class LassoRegression() :
      
    def __init__( self, learning_rate, iterations, l1_penality ) :
          
        self.learning_rate = learning_rate
          
        self.iterations = iterations
          
        self.l1_penality = l1_penality

        self.time_iter_grad = []

        self.objval_grad = []
    # Function for model training
              
    def fit( self, X, Y ) :
          
        # no_of_training_examples, no_of_features
          
        self.m, self.n = X.shape
          
        # weight initialization
          
        self.W = np.zeros( self.n )
          
        self.X = X
          
        self.Y = Y
        
        # gradient descent learning
                  
        for i in range( self.iterations ) :
            pred = self.predict( self.X )
            h1 = self.objval(Y, pred , X)
            start_time_grad = t.time()
            self.update_weights()
            time_grad = round(t.time()- start_time_grad,7)
            self.time_iter_grad.append(time_grad)
            pred = self.predict( self.X )
            h2 = self.objval(Y, pred , X)
            self.objval_grad.append(h2)
            if(abs(h2-h1)<=0.0001):
              break
        time_g = (self.time_iter_grad)
        time_iterg = time_g.copy()
        time_iterg.sort(reverse=True)
        self.time_iter_grad = time_iterg
        return self, self.time_iter_grad
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :
             
        Y_pred = self.predict( self.X )
          
        # calculate gradients  
          
        dW = np.zeros( self.n )
          
        for j in range( self.n ) :
              
            if self.W[j] > 0 :
                  
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) ) 
                           
                         + self.l1_penality ) / self.m
          
            else :
                  
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) ) 
                           
                         - self.l1_penality ) / self.m
  
          
        # update weights
      
        self.W = self.W - self.learning_rate * dW
          
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :
        return X.dot( self.W ) 

    def objval(self ,Y,pred,X):
      h = (sum( (Y - pred )**2) + self.l1_penality * la.norm(self.W, 1) ) / self.m
      return h