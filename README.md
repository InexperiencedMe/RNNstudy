# RNNstudy
Manual RNN in Numpy and simpler equivalent in PyTorch

The task solved is binary addition, bit by bit, which is a very simple problem, that's impossible to do for a simple multilayer perceptron, as it requires remembering the recent past bits to correctly process the sequence.

The whole inspiration came from Andrew Trask and his [blog post](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/), that I went through, analyzed the code line by line, updated the code to more modern standards and did the same in PyTorch to have a reference of how to use it there with an autograd engine.
