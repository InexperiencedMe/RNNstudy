import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

numBits         = 8
largestNumber   = 2 ** numBits
inputDim        = 2
hiddenDim       = 16
outputDim       = 1
learningRate    = 0.1
numIterations   = 10000

class recurrentBinaryAdder(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(recurrentBinaryAdder, self).__init__()
        self.hiddenDim = hiddenDim
        self.gru = nn.GRU(inputDim, hiddenDim)
        self.linear = nn.Linear(hiddenDim, outputDim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input, hiddenState):
        hiddenOut, newHiddenState = self.gru(input, hiddenState)
        finalOutput = self.sigmoid(self.linear(hiddenOut))
        return finalOutput, newHiddenState
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenDim)

def binToInt(binary_tensor):
    return sum([int(binary_tensor[i].item()) * (2 ** (numBits - 1 - i)) for i in range(numBits)])

def intToBin(n, numBits=numBits):
    binaryString = bin(n)[2:].zfill(numBits)
    return torch.tensor([int(b) for b in binaryString], dtype=torch.float32) # PyTorch demands float32

formatTensorForPrinting = lambda tensor: str(tensor.int().tolist()).replace(",", "")


model = recurrentBinaryAdder(inputDim, hiddenDim, outputDim)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
criterion = nn.MSELoss()
for iteration in range(numIterations):
    a_int = np.random.randint(largestNumber // 2)
    b_int = np.random.randint(largestNumber // 2)
    c_int = a_int + b_int
    
    a_bin = intToBin(a_int)
    b_bin = intToBin(b_int)
    c_bin = intToBin(c_int)
    d_bin = torch.zeros_like(c_bin)
    
    hidden = model.initHidden()
    optimizer.zero_grad()
    totalLoss = 0
    for position in range(numBits - 1, -1, -1):
        x = torch.tensor([[a_bin[position], b_bin[position]]]).view(1, 1, inputDim)
        y = c_bin[position].view(1, 1, outputDim)

        output, hidden = model(x, hidden)
        partialLoss = criterion(output, y)
        totalLoss += partialLoss
        
        d_bin[position] = torch.round(output)
    
    totalLoss.backward()
    optimizer.step()
    
    
    if(iteration % 1000 == 0):
        print(f"Error: {totalLoss:.4f}")
        print(f"Ground truth : {formatTensorForPrinting(c_bin)}")
        print(f"Model output : {formatTensorForPrinting(d_bin)}")
        print(f"{a_int} + {b_int} = {binToInt(d_bin)}")
        print("------------")