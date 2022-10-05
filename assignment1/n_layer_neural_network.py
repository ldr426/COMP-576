from three_layer_neural_network import NeuralNetwork
import numpy as np

class Layer():
    '''
    Build a single layer
    '''
    def __init__(self, input_dim, output_dim, actFun, diff_actFun, seed=0, *args, **kwargs):
        '''
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param actFun: activation function
        :param diff_actFun: derivative of activation function
        :param seed: random seed
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun = actFun
        self.diff_actFun = diff_actFun

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self.output_dim))
    
    def feedforward(self, X):
        '''
        feed X
        '''
        self.X = X
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z)
        return self.a
    
    def backprop(self, da):
        '''
        calculate relative para
        '''
        num_examples = len(self.X)
        self.dW = self.X.T.dot(da * (self.diff_actFun(self.z)))
        self.db = np.ones(num_examples).dot(da * (self.diff_actFun(self.z)))
        self.dX = (da * self.diff_actFun(self.z)).dot(self.W.T)
        
def fix_type(fn, type):
    '''
    Fix the type parameter
    '''
    def wrapper(*args, **kwargs):
        return fn(type=type, *args, **kwargs)
    return wrapper


class DeepNeuralNetwrok(NeuralNetwork):
    
    def __init__(self, n_hidden, input_dim, hidden_dim, output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param n_hidden: number of hidden layers
        :param size_hidden: size of hidden layers
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.seed = seed

        # initialize the layers
        self.layers = [Layer(self.input_dim, self.hidden_dim, fix_type(self.actFun, actFun_type), fix_type(self.diff_actFun, actFun_type))]
        for _ in range(self.n_hidden):
            self.layers += [Layer(hidden_dim, hidden_dim, fix_type(self.actFun, actFun_type), fix_type(self.diff_actFun, actFun_type))]

        # initialize output layer weight and bias
        self.W_out = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.b_out = np.zeros((1, self.output_dim))
        
    def feedforward(self, X):
        for layer in self.layers:
            X = layer.feedforward(X)
        
        self.z_out = X.dot(self.W_out) + self.b_out
        self.probs = self.softmax(self.z_out)
    
    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        '''
        # calculate the dw for the last layer
        y_onehot = np.stack((1-y, y), -1)
        self.dW_out = self.layers[-1].a.T.dot(self.probs - y_onehot)
        da = (self.probs - y_onehot).dot(self.W_out.T)

        for layer in reversed(self.layers):
            layer.backprop(da)
            da = layer.dX
        
    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X)
        
        # Calculating the loss
        y_onehot = np.stack((1-y, y), -1)
        data_loss = -(y_onehot * np.log(self.probs)).sum()

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(np.concatenate([layer.W.ravel() for layer in self.layers]))))
        return (1. / num_examples) * data_loss


    def fit_model(self, X, y, eps=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):

            self.feedforward(X)
            self.backprop(X, y)
            
            self.dW_out += self.reg_lambda * self.W_out
            for layer in self.layers:
                layer.dW += self.reg_lambda * layer.W

            # Gradient descent parameter update
            self.W_out += -eps * self.dW_out
            for layer in self.layers:
                layer.W += -eps * layer.dW
                layer.b += -eps * layer.db