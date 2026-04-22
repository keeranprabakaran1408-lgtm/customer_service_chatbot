import random
import json
import torch
from model_for_chatbot import NeuralNet
from nltk_file import Chatbot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("test_data.json", 'r') as f:
    company_file_test_data = json.load(f)


myFile = "data.pth"
data = torch.load(myFile)
input_size = data["input_size"]#define the dictionary here for training of data
output_size = data["output_size"]
hidden_size = data["hidden_size"]
tokens = data["tokens"]
tokens_test_data_labels = data["tokens_test_data_labels"]
chatbot_model_state = data["chatbot_model_state"]
chatbot_model = NeuralNet(input_size, hidden_size, output_size).to(device)
chatbot_model.load_state_dict(chatbot_model_state)#load dictionary
chatbot_model.eval()

bot_name = "Customer Service Chatbot"
print("Hi, how can I help? (type exit when finished)")

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break

    user_input = Chatbot.tokenize(user_input)#tokenize user_input
    X = Chatbot.bag_of_words(user_input, tokens)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = chatbot_model(X)
    _, predicted = torch.max(output, dim=1)

    tokens_test_data_labels1 = tokens_test_data_labels[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for test_data in company_file_test_data['test_data']:
            if tokens_test_data_labels1 == test_data["type_of_test_data"]:
                print(f"{bot_name}: {random.choice(test_data['responses'])}")
    else:
        print(f"{bot_name}: I don't understand please type your input again")

