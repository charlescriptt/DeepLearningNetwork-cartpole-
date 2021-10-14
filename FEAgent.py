import gym
from collections import deque
import numpy as np


def relu(inp):
    return np.multiply(inp,(inp>0))
    
def relu_derivative(inp):
    return (inp>0)*1

class NNLayer:
    # class representing a neural net layer
    def __init__(self, inSize, outSize, activation=None, lr = 0.001):
        #here we have intitialised all inputs required to implement our nueral network
        #number of inputs from the observation space (5 for cartpole incl bias)
        self.inSize = inSize
        #number of output actions (L, R)
        self.outSize = outSize
        #initialy we will set randomized weights to soon tune
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(inSize, outSize))
        self.actFunction = activation
        self.lr = lr

    # Computes the forward pass for this nn layer
    def forward(self, inputs, remember_for_backprop=True):
        #first we add a bias onto our inputs
        inpWithBias = np.append(inputs,1)
        #we then calculate product of input and weight matrix
        unactivated = np.dot(inpWithBias, self.weights)
        output = unactivated
        if self.actFunction != None:
            output = self.actFunction(output)
        if remember_for_backprop:
            self.backwardIn = inpWithBias
            self.backwardOut = np.copy(unactivated)
        return output    
        
    def update_weights(self, gradient):
        #gradiant calculated from backwards, loss function derivitive
        self.weights = self.weights - self.lr*gradient
        
    def backward(self, gradient):
    
        #if there is an activation function
        if self.actFunction != None:
            #pointwise we multiply the derivative of activation
            multiplier = np.multiply(relu_derivative(self.backwardOut),gradient)
        else:
            #else we initialise multiplier as gradiant
            multiplier = gradient
        
        #loss function derivative with respect to weights, sent to update_weights
        lossGrad = np.dot(np.transpose(np.reshape(self.backwardIn, (1, len(self.backwardIn)))), np.reshape(multiplier, (1,len(multiplier))))
        #calculated error of inputs in this layer, which is in turn passed to previous layer for further calculation
        inError = np.dot(multiplier, np.transpose(self.weights))[:-1]
        self.update_weights(lossGrad)
        return inError

class FEA:
    # class representing a reinforcement learning agent
    env = None
    def __init__(self, env):
        self.env = env
        
        # 2 hidden layers, initialisation of layers, input and number
        self.hiddenLayers = 2
        self.hiddenSize = 24
        self.outputSize = env.action_space.n
        self.inputSize = env.observation_space.shape[0]  
        
        
        #holds past experiences in memory, deque allows for saving and appending recent memory
        self.mem = deque([],1000000)
        self.gamma = 0.95
        self.eps = 1.0
        
        self.layers = [NNLayer(self.inputSize + 1, self.hiddenSize, activation=relu)]
        for i in range(self.hiddenLayers-1):
            self.layers.append(NNLayer(self.hiddenSize+1, self.hiddenSize, activation=relu))
        self.layers.append(NNLayer(self.hiddenSize+1, self.outputSize))
        
    def select_action(self, observation):
        #using the forward function values will become the action with highest percieved reward
        values = self.forward(np.asmatrix(observation))
        #below allows a balance between exploration vs exploitation to ensure we dont get lost in local minima
        if (np.random.random() > self.eps):
            return np.argmax(values)
        else:
            return np.random.randint(self.env.action_space.n)
            
    def forward(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        #loop for proceduly going through layers, collecting outputs and passing to the next
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
            index = index + 1
        return vals
        
    def remember(self, done, action, observation, prev_obs):
        self.mem.append([done, action, observation, prev_obs])
        
    def experience_replay(self, update_size=20):
        #first we check if there is enough information stored to work with 
        if (len(self.mem) < update_size):
            return
        else: 
            #when we have enough data we then randomly select samples from memory (no. of samples = update size)
            batch_indices = np.random.choice(len(self.mem), update_size)
            for index in batch_indices:
                #for each of these selected indices we store their data
                done, action_selected, new_obs, prev_obs = self.mem[index]
                #from this data we calculate 3 key variables below:
                #action value gathered from prev observation
                #works as an estimate of value of each action in our current state
                actionVals = self.forward(prev_obs, remember_for_backprop=True)
                #next action is calculated observation and is about next moves, used for calculating experimental when needed
                nActionVals = self.forward(new_obs, remember_for_backprop=False)
                #initialise experimental values for use if needed
                expVals = np.copy(actionVals)
                if done:
                    #pole has tipped over, so all rewards past here are -1
                    expVals[action_selected] = -1
                else:
                    #pole is still standing, +1 for still standing multiplied by gamma of best percieved reward
                    expVals[action_selected] = 1 + self.gamma*np.max(nActionVals)
                #we then pass our calculated values to backward function to alter weights appropraitely
                self.backward(actionVals, expVals)
        #now we update the epsilon value in accordance with needed learning rate, stops learning dropping below certain rate
        self.eps = self.eps if self.eps < 0.01 else self.eps*0.997
        for layer in self.layers:
            layer.lr = layer.lr if layer.lr < 0.0001 else layer.lr*0.99
        
    def backward(self, actionVals, expVals): 
        # difference between calculated and experimental_values calculated
        diff = (actionVals - expVals)
        # this differance is then propogated backwards through each layer and weights are updated
        #effectively each layer calculates how much their ourput differs frome expected, which would be needed for experimental_values
        for layer in reversed(self.layers):
            #calls nn layer function
            diff = layer.backward(diff)