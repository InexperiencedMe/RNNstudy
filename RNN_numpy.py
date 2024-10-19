import copy, numpy as np
np.random.seed(0)

numBits         = 8
largestNumber   = 2 ** numBits
learningRate    = 0.1
inputDim        = 2
hiddenDim       = 16
outputDim       = 1
numIterations   = 10000

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidDerivative(x):
    return x*(1 - x)

def intToBin(n, numBits=numBits):
    binaryString = bin(n)[2:].zfill(numBits)
    return np.array([int(b) for b in binaryString], dtype=np.uint8) # bools print True and False, so using uint8

def binToInt(binaryArray):
    binaryString = ''.join(map(str, binaryArray))
    return int(binaryString, 2)


# 2*[0.0, 1.0) - 1 to make it [-1.0, 1) range
weights_01 = 2*np.random.random((inputDim, hiddenDim)) - 1
weights_11 = 2*np.random.random((hiddenDim, hiddenDim)) - 1
weights_12 = 2*np.random.random((hiddenDim, outputDim)) - 1

weights_01_update = np.zeros_like(weights_01)
weights_11_update = np.zeros_like(weights_11)
weights_12_update = np.zeros_like(weights_12)

for iteration in range(numIterations):
    a_int = np.random.randint(largestNumber/2)
    b_int = np.random.randint(largestNumber/2)
    c_int = a_int + b_int

    a_bin = intToBin(a_int)         # number A to add
    b_bin = intToBin(b_int)         # number B to add
    c_bin = intToBin(c_int)         # number C as a ground truth result
    d_bin = np.zeros_like(c_bin)    # number D as our current network output

    overallError        = 0
    layer_2_gradients   = [] 
    layer_1_values      = [np.zeros(hiddenDim)] # recurrent state values initialized with one zeroed state
    
    for position in range(numBits - 1, -1, -1):
        x = np.array([[a_bin[position], b_bin[position]]])
        y = np.array([[c_bin[position]]])

        layer_1 = sigmoid(np.dot(x, weights_01) + np.dot(layer_1_values[-1], weights_11))
        layer_2 = sigmoid(np.dot(layer_1, weights_12))

        layer_2_partialError = y - layer_2 # Derivative of mean squared error (1/2) * (y - output)^2
        layer_2_gradients.append((layer_2_partialError)*sigmoidDerivative(layer_2))
        overallError += np.abs(layer_2_partialError[0]).item()

        d_bin[position] = np.round(layer_2.item())
        layer_1_values.append(copy.deepcopy(layer_1))
    

    future_layer_1_gradient = np.zeros(hiddenDim)

    for position in range(numBits):
        x = np.array([[a_bin[position], b_bin[position]]])

        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]
        layer_2_gradient = layer_2_gradients[-position - 1]
        layer_1_gradient = (future_layer_1_gradient.dot(weights_11.T) + layer_2_gradient.dot(weights_12.T)) * sigmoidDerivative(layer_1)

        weights_12_update += np.atleast_2d(layer_1).T.dot(layer_2_gradient)
        weights_11_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_gradient)
        weights_01_update += x.T.dot(layer_1_gradient)
        
        future_layer_1_gradient = layer_1_gradient
    

    weights_01 += weights_01_update*learningRate
    weights_11 += weights_11_update*learningRate    
    weights_12 += weights_12_update*learningRate

    weights_01_update *= 0
    weights_11_update *= 0
    weights_12_update *= 0
    
    if(iteration % 1000 == 0):
        print(f"Error: {overallError:.4f}")
        print(f"Ground truth : {str(c_bin):4}")
        print(f"Model output : {str(d_bin):4}")
        print(f"{a_int} + {b_int} = {binToInt(d_bin)}")
        print("------------")