import numpy as np

class RecurrentNeuralNetwork:
    #input (in t), expected output (shifted by one time step), num of words (num of recurrences), array of expected outputs, learning rate

    def __init__ (self, input, output, recurrences, expected_output, learning_rate):
        #initial input
        self.x = np.zeros(input)
        #input size
        self.input = input
        #expected output (shifted by one time step)
        self.y = np.zeros(output)
        #output size
        self.output = output
        #weight matrix for interpreting results from hidden state cell (num words x num words matrix)
        #random initialization is crucial here
        self.w = np.random.random((output, output))
        #matrix used in RMSprop in order to decay the learning rate
        self.G = np.zeros_like(self.w)
        #length of the recurrent network - number of recurrences i.e num of inputs
        self.recurrences = recurrences
        #learning rate
        self.learning_rate = learning_rate
        #array for storing inputs
        self.ia = np.zeros((recurrences+1,input))
        #array for storing cell states
        self.ca = np.zeros((recurrences+1,output))
        #array for storing outputs
        self.oa = np.zeros((recurrences+1,output))
        #array for storing hidden states
        self.ha = np.zeros((recurrences+1,output))
        #forget gate
        self.af = np.zeros((recurrences+1,output))
        #input gate
        self.ai = np.zeros((recurrences+1,output))
        #cell state
        self.ac = np.zeros((recurrences+1,output))
        #output gate
        self.ao = np.zeros((recurrences+1,output))
        #array of expected output values
        self.expected_output = np.vstack((np.zeros(expected_output.shape[0]), expected_output.T))
        #declare LSTM cell (input, output, amount of recurrence, learning rate)
        self.LSTM = LSTM(input, output, recurrences, learning_rate)

    #activation function. simple nonlinearity, converts influx floats into probabilities between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #the derivative of the sigmoid function. We need it to compute gradients for backpropagation
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

        #Here is where magic happens: We apply a series of matrix operations to our input and compute an output
    def forwardProp(self):
        for i in range(1, self.recurrences+1):
            self.LSTM.x = np.hstack((self.ha[i-1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            #store cell state from the forward propagation
            self.ca[i] = cs #cell state
            self.ha[i] = hs #hidden state
            self.af[i] = f #forget state
            self.ai[i] = inp #inpute gate
            self.ac[i] = c #cell state
            self.ao[i] = o #output gate
            self.oa[i] = self.sigmoid(np.dot(self.w, hs)) #activate the weight*input
            self.x = self.expected_output[i-1]
        return self.oa


    def backProp(self):
        #Backpropagation of our weight matrices (Both in our Recurrent network + weight matrices inside LSTM cell)
        #start with an empty error value
        totalError = 0
        #initialize matrices for gradient updates
        #First, these are RNN level gradients
        #cell state
        dfcs = np.zeros(self.output)
        #hidden state,
        dfhs = np.zeros(self.output)
        #weight matrix
        tu = np.zeros((self.output,self.output))
        #Next, these are LSTM level gradients
        #forget gate
        tfu = np.zeros((self.output, self.input+self.output))
        #input gate
        tiu = np.zeros((self.output, self.input+self.output))
        #cell unit
        tcu = np.zeros((self.output, self.input+self.output))
        #output gate
        tou = np.zeros((self.output, self.input+self.output))
        #loop backwards through recurrences
        for i in range(self.recurrences, -1, -1):
            #error = calculatedOutput - expectedOutput
            error = self.oa[i] - self.expected_output[i]
            #calculate update for weight matrix
            #Compute the partial derivative with (error * derivative of the output) * hidden state
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)
            #Now propagate the error back to exit of LSTM cell
            #1. error * RNN weight matrix
            error = np.dot(error, self.w)
            #2. set input values of LSTM cell for recurrence i (horizontal stack of array output, hidden + input)
            self.LSTM.x = np.hstack((self.ha[i-1], self.ia[i]))
            #3. set cell state of LSTM cell for recurrence i (pre-updates)
            self.LSTM.cs = self.ca[i]
            #Finally, call the LSTM cell's backprop and retreive gradient updates
            #Compute the gradient updates for forget, input, cell unit, and output gates + cell states + hiddens states
            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(error, self.ca[i-1], self.af[i], self.ai[i], self.ac[i], self.ao[i], dfcs, dfhs)
            #Accumulate the gradients by calculating total error (not necesarry, used to measure training progress)
            totalError += np.sum(error)
            #forget gate
            tfu += fu
            #input gate
            tiu += iu
            #cell state
            tcu += cu
            #output gate
            tou += ou
        #update LSTM matrices with average of accumulated gradient updates
        self.LSTM.update(tfu/self.recurrences, tiu/self.recurrences, tcu/self.recurrences, tou/self.recurrences)
        #update weight matrix with average of accumulated gradient updates
        self.update(tu/self.recurrences)
        #return total error of this iteration
        return totalError

    def update(self, u):
        #Implementation of RMSprop in the vanilla world
        #We decay our learning rate to increase convergence speed
        self.G = 0.95 * self.G + 0.1 * u**2
        self.w -= self.learning_rate/np.sqrt(self.G + 1e-8) * u
        return

    #We define a sample function which produces the output once we trained our model
    #let's say that we feed an input observed at time t, our model will produce an output that can be
    #observe in time t+1
    def sample(self):
        #loop through recurrences - start at 1 so the 0th entry of all output will be an array of 0's
        for i in range(1, self.recurrences+1):
            #set input for LSTM cell, combination of input (previous output) and previous hidden state
            self.LSTM.x = np.hstack((self.ha[i-1], self.x))
            #run forward prop on the LSTM cell, retrieve cell state and hidden state
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            #store input as vector
            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x #Use np.argmax?
            #store cell states
            self.ca[i] = cs
            #store hidden state
            self.ha[i] = hs
            #forget gate
            self.af[i] = f
            #input gate
            self.ai[i] = inp
            #cell state
            self.ac[i] = c
            #output gate
            self.ao[i] = o
            #calculate output by multiplying hidden state with weight matrix
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            #compute new input
            maxI = np.argmax(self.oa[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX
        #return all outputs
        return self.oa

class LSTM:
    # LSTM cell (input, output, amount of recurrence, learning rate)
    def __init__ (self, input, output, recurrences, learning_rate):
        #input size
        self.x = np.zeros(input+output)
        #input size
        self.input = input + output
        #output
        self.y = np.zeros(output)
        #output size
        self.output = output
        #cell state intialized as size of prediction
        self.cs = np.zeros(output)
        #how often to perform recurrence
        self.recurrences = recurrences
        #balance the rate of training (learning rate)
        self.learning_rate = learning_rate
        #init weight matrices for our gates
        #forget gate
        self.f = np.random.random((output, input+output))
        #input gate
        self.i = np.random.random((output, input+output))
        #cell state
        self.c = np.random.random((output, input+output))
        #output gate
        self.o = np.random.random((output, input+output))
        #forget gate gradient
        self.Gf = np.zeros_like(self.f)
        #input gate gradient
        self.Gi = np.zeros_like(self.i)
        #cell state gradient
        self.Gc = np.zeros_like(self.c)
        #output gate gradient
        self.Go = np.zeros_like(self.o)

    #activation function. simple nonlinearity, converts influx floats into probabilities between 0 and 1
    #same as in rnn class
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #The derivativeative of the sigmoid function. We need it to compute gradients for backpropagation
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    #Within the LSTM cell we are going to use a tanh activation function.
    #Long story short: This leads to stronger gradients (derivativeatives are higher) which is usefull as our data is centered around 0,
    def tangent(self, x):
        return np.tanh(x)

    #derivativeative for computing gradients
    def dtangent(self, x):
        return 1 - np.tanh(x)**2

    #Here is where magic happens: We apply a series of matrix operations to our input and compute an output
    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o


    def backProp(self, e, pcs, f, i, c, o, dfcs, dfhs):
        #error = error + hidden state derivativeative. we clip the value between -6 and 6 to prevent vanishing
        e = np.clip(e + dfhs, -6, 6)
        #multiply error by activated cell state to compute output derivativeative
        do = self.tangent(self.cs) * e
        #output update = (output derivative * activated output) * input
        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        #derivativeative of cell state = error * output * derivativeative of cell state + derivative cell
        #compute gradients and update them in reverse order!
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        #derivativeative of cell = derivativeative cell state * input
        dc = dcs * i
        #cell update = derivativeative cell * activated cell * input
        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        #derivativeative of input = derivative cell state * cell
        di = dcs * c
        #input update = (derivative input * activated input) * input
        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))
        #derivative forget = derivative cell state * all cell states
        df = dcs * pcs
        #forget update = (derivative forget * derivative forget) * input
        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))
        #derivative cell state = derivative cell state * forget
        dpcs = dcs * f
        #derivative hidden state = (derivative cell * cell) * output + derivative output * output * output derivative input * input * output + derivative forget
        #* forget * output
        dphs = np.dot(dc, self.c)[:self.output] + np.dot(do, self.o)[:self.output] + np.dot(di, self.i)[:self.output] + np.dot(df, self.f)[:self.output]
        #return update gradients for forget, input, cell, output, cell state, hidden state
        return fu, iu, cu, ou, dpcs, dphs

    def update(self, fu, iu, cu, ou):
        #Update forget, input, cell, and output gradients
        self.Gf = 0.9 * self.Gf + 0.1 * fu**2
        self.Gi = 0.9 * self.Gi + 0.1 * iu**2
        self.Gc = 0.9 * self.Gc + 0.1 * cu**2
        self.Go = 0.9 * self.Go + 0.1 * ou**2

        #Update our gates using our gradients
        self.f -= self.learning_rate/np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.learning_rate/np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.learning_rate/np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.learning_rate/np.sqrt(self.Go + 1e-8) * ou
        return
def read_text_file():
    #Open textfile and return a separated input and output variable(series of words)
    with open("/home/johnny/IdeaProjects/RNN/resources/test.txt", "r") as text_file:
        data = text_file.read()
    text = list(data)
    outputSize = len(text)
    data = list(set(text))
    uniqueWords, dataSize = len(data), len(data)
    returnData = np.zeros((uniqueWords, dataSize))
    for i in range(0, dataSize):
        returnData[i][i] = 1
    returnData = np.append(returnData, np.atleast_2d(data), axis=0)
    output = np.zeros((uniqueWords, outputSize))
    for i in range(0, outputSize):
        index = np.where(np.asarray(data) == text[i])
        output[:,i] = returnData[0:-1,index[0]].astype(float).ravel()
    return returnData, uniqueWords, output, outputSize, data

#Write the predicted output (series of words) to disk
def export_to_textfile(output, data):
    finalOutput = np.zeros_like(output)
    prob = np.zeros_like(output[0])
    outputText = ""
    print(len(data))
    print(output.shape[0])
    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            prob[j] = output[i][j] / np.sum(output[i])
        outputText += np.random.choice(data, p=prob)
    with open("out.txt", "w") as text_file:
        text_file.write(outputText)
    return

print("Initialize Hyperparameters")
iterations = 10000
learningRate = 0.001
print("Reading Inputa/Output data from disk")
returnData, numCategories, expectedOutput, outputSize, data = read_text_file()
print("Reading from disk done. Proceeding with training.")
#Initialize the RNN using our hyperparameters and data
RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)

#training time!
for i in range(1, iterations):
    #Predict the next word of each word
    RNN.forwardProp()
    #update all our weights using our error
    error = RNN.backProp()
    #For a given error threshold
    print("Reporting error on iteration ", i, ": ", error)
    if error > -10 and error < 10 or i % 10 == 0:
        #We provide a seed word
        seed = np.zeros_like(RNN.x)
        maxI = np.argmax(np.random.random(RNN.x.shape))
        seed[maxI] = 1
        RNN.x = seed
        #and predict the upcoming one
        output = RNN.sample()
        print(output)
        #finally, we store it to disk
        export_to_textfile(output, data)
        print("Done Writing")
print("Train/Prediction routine complete.")