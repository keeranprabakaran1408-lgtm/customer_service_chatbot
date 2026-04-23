# customer_service_chatbot
A customer service chatbot I built in Python for my A-level Computer Science project. It uses a neural network I trained from scratch to understand and respond to customer queries.

## What it does

- Built and trained a 3-layer neural network using PyTorch to classify customer queries
- Used NLTK for tokenization and stemming to process user input before passing it to the model
- Trained the model using a bag of words approach with cross entropy loss over 1000 epochs
- Saves the trained model to a file so it doesn't need retraining every time it runs
- Pygame GUI with a chat window, input box and send button
- Uses Microsoft's DialoGPT model to generate responses
- Responds with confidence — if the model isn't confident enough (below 75%) it asks the user to rephrase

## Files

- `model_for_chatbot.py` — defines the 3-layer neural network architecture
- `train_model.py` — trains the model on the dataset and saves it to `data.pth`
- `nltk_file.py` — handles tokenization, stemming and bag of words
- `user_interface.py` — loads the trained model and runs the chatbot in the terminal
- `chatbot.py` — Pygame GUI version of the chatbot
- `test_data.json` — the dataset used to train the model (not included)

## Built with

- Python
- PyTorch
- NLTK
- Pygame
- Transformers
- NumPy

## How to run

1. Make sure Python is installed — https://www.python.org
2. Install the required libraries:
```
pip install pygame torch transformers nltk numpy
```
3. Train the model first:
```
python train_model.py
```
4. Then run the chatbot:
```
python user_interface.py
```
Or for the GUI version:
```
python chatbot.py
```
