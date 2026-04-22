import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):# num_classes is the output layer of the neural networks class
        super(NeuralNet, self).__init__()#makes the class into superclass for use
        self.linear_layer1 = nn.Linear(input_size, hidden_size)#input_size is used as input
        self.linear_layer2 = nn.Linear(hidden_size, hidden_size)
        self.linear_layer3 = nn.Linear(hidden_size, num_classes)#num_classes used as output
        self.relu = nn.ReLU()#activation function to be used in between the different layers


    def forward(self, x):#forward function for loss/weight of the model
        out = self.linear_layer1(x)#x represents the stage numbers for process of layering for inputs
        out = self.relu(out)#out used as template variable for output of layers
        out = self.linear_layer2(out)
        out = self.relu(out)
        out = self.linear_layer3(out)
        return out

