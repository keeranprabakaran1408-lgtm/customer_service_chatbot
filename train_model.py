import json
import numpy as np
from nltk_file import Chatbot

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_for_chatbot import NeuralNet

with open("test_data.json", 'r') as f:#file handling opening & reading test_data file
    company_file_test_data = json.load(f)#variable to load test_data file


tokens = []#empty array to continuously append all the data stored in test_data.json
tokens_test_data_labels = []#array -class labels for file and type of test data used
similar_inputs_for_tokens = []#array -common patterns for user inputs

for test_data in company_file_test_data['test_data']:#iterate every class label
    type_of_test_data = test_data["type_of_test_data"]
    tokens_test_data_labels.append(type_of_test_data)
    for similar_input in test_data['similar_inputs']:#iterate every pattern
        word_tokenization_for_similar_inputs = Chatbot.tokenize(similar_input)
        tokens.extend(word_tokenization_for_similar_inputs)
        similar_inputs_for_tokens.append((word_tokenization_for_similar_inputs, type_of_test_data))


characters = ['!', '.', ';', ',', '?', '/',]#punctuation removed
tokens = [Chatbot.stem(i) for i in tokens if i not in characters]#iterate to remove punctuation
tokens = sorted(set(tokens))
tokens_test_data_labels = sorted(set(tokens_test_data_labels))

labels_bag_of_words = []#for label/type_of_test_data that iterates similar_inputs_for_tokens array
num_of_labels = []#index for each data item in the tokens array
for(similar_input_sentence, type_of_test_data) in similar_inputs_for_tokens:
    bag = Chatbot.bag_of_words(similar_input_sentence, tokens)
    labels_bag_of_words.append(bag)


    labels = tokens_test_data_labels.index(type_of_test_data)
    num_of_labels.append(labels)



labels_bag_of_words = np.array(labels_bag_of_words)#implement the training of data for responses
num_of_labels = np.array(num_of_labels)



class ChatbotDataset(Dataset):#had to create an additional class to implement the dataset for the program
    def __init__(self):
        self.n_sample_test_data = len(labels_bag_of_words)
        self.test_data_for_bag_of_words = labels_bag_of_words
        self.num_of_labels_test_data = num_of_labels


    def __getitem__(self, index):#index for training test data
        return self.test_data_for_bag_of_words[index], self.num_of_labels_test_data[index]

    def __len__(self):
        return self.n_sample_test_data

input_size = len(labels_bag_of_words[0])#length of data stored in the tokens array to be passed in NeuralNet class
output_size = len(tokens_test_data_labels)
hidden_size = 8
num_epochs = 1000 #number of times the neural networks see the dataset whilst training

dataset = ChatbotDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)#loading and processing data for training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#have to change to same device for pytorch
chatbot_model = NeuralNet(input_size, hidden_size, output_size).to(device)#NeuralNet class called from model_for_chatbot.py

criterion = nn.CrossEntropyLoss()#measure probability of responses being generated
optimizer = torch.optim.Adam(chatbot_model.parameters(), lr=0.001)#lr how frequent model is updated

for epoch in range(num_epochs):#iterates for each time the neural networks see the dataset
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        outputs = chatbot_model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:#completleting the total number of epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')#print the weight of each epoch
print(f'final loss: {loss.item():.4f}')


data = {
        "chatbot_model_state":chatbot_model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "tokens": tokens,
        "tokens_test_data_labels": tokens_test_data_labels
}

myFile = "data.pth"#file saved and created to keep on updating/training after every input
torch.save(data, myFile)
print(f'file saved to {myFile}')
