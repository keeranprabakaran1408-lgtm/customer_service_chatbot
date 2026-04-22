import random
import json
import torch
import pygame
import sys
import pyaudio  # For recording audio from microphone input
import speech_recognition as sr  # For converting digital signals of speech to text
from model_for_chatbot import NeuralNet
from nltk_file import Chatbot

# Initialize Pygame
pygame.init()
font = pygame.font.Font(None, 30)
screen = pygame.display.set_mode((1200, 1000))
screen.fill((0, 0, 0))
clock = pygame.time.Clock()

# Load chatbot model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("test_data.json", 'r') as f:
    company_file_test_data = json.load(f)

myFile = "data.pth"
data = torch.load(myFile)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
tokens = data["tokens"]
tokens_test_data_labels = data["tokens_test_data_labels"]
chatbot_model_state = data["chatbot_model_state"]
chatbot_model = NeuralNet(input_size, hidden_size, output_size).to(device)
chatbot_model.load_state_dict(chatbot_model_state)
chatbot_model.eval()

bot_name = "Customer Service Chatbot"
print("Hi, how can I help? (type exit when finished)")

# Keyboard class
class Keyboard:
    def __init__(self, text, width, height, pos, elevation):
        self.text = text
        self.pressed = False
        self.elevation = elevation
        self.change_in_elevation = elevation
        self.original_yposition = pos[1]
        self.top_rectangle = pygame.Rect(pos, (width, height))
        self.top_colour_for_rectangle = '#354661'
        self.bottom_rectangle = pygame.Rect(pos, (width, height))
        self.bottom_colour_for_rectangle = '#e8eaed'
        self.text_surface = font.render(text, True, '#f0f0f0')
        self.text_rectangle = self.text_surface.get_rect(center=self.top_rectangle.center)

    def design_of_keyboard_keys(self):
        self.top_rectangle.y = self.original_yposition - self.change_in_elevation
        self.text_rectangle.center = self.top_rectangle.center
        self.bottom_rectangle.midtop = self.top_rectangle.midtop
        pygame.draw.rect(screen, self.bottom_colour_for_rectangle, self.bottom_rectangle, border_radius=8)
        pygame.draw.rect(screen, self.top_colour_for_rectangle, self.top_rectangle, border_radius=8)
        screen.blit(self.text_surface, self.text_rectangle)
        return self.mouse_position_of_keyboard()

    def mouse_position_of_keyboard(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rectangle.collidepoint(mouse_pos):
            self.top_colour_for_rectangle = '#cacbcc'
            if pygame.mouse.get_pressed()[0]:
                self.change_in_elevation = 0
                self.pressed = True
                return self.text
            else:
                self.change_in_elevation = self.elevation
                if self.pressed:
                    self.pressed = False
        else:
            self.change_in_elevation = self.elevation
            self.top_colour_for_rectangle = '#354661'

# Button class
class Button:
    def __init__(self, text, width, height, pos, elevation):
        self.text = text
        self.pressed = False
        self.elevation = elevation
        self.change_in_elevation = elevation
        self.original_yposition = pos[1]
        self.top_rectangle = pygame.Rect(pos, (width, height))
        self.top_colour_for_rectangle = '#354661'
        self.bottom_rectangle = pygame.Rect(pos, (width, height))
        self.bottom_colour_for_rectangle = '#e8eaed'
        self.text_surface = font.render(text, True, '#f0f0f0')
        self.text_rectangle = self.text_surface.get_rect(center=self.top_rectangle.center)

    def design_of_buttons(self):
        self.top_rectangle.y = self.original_yposition - self.change_in_elevation
        self.text_rectangle.center = self.top_rectangle.center
        self.bottom_rectangle.midtop = self.top_rectangle.midtop
        pygame.draw.rect(screen, self.bottom_colour_for_rectangle, self.bottom_rectangle, border_radius=8)
        pygame.draw.rect(screen, self.top_colour_for_rectangle, self.top_rectangle, border_radius=8)
        screen.blit(self.text_surface, self.text_rectangle)
        return self.mouse_position_of_buttons()

    def mouse_position_of_buttons(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rectangle.collidepoint(mouse_pos):
            self.top_colour_for_rectangle = '#cacbcc'
            if pygame.mouse.get_pressed()[0]:
                self.change_in_elevation = 0
                self.pressed = True
                return self.text
            else:
                self.change_in_elevation = self.elevation
                if self.pressed:
                    self.pressed = False
        else:
            self.change_in_elevation = self.elevation
            self.top_colour_for_rectangle = '#354661'

# Initialize keyboard keys
key_positions = [
    ('Q', (100, 550)), ('W', (160, 550)), ('E', (220, 550)), ('R', (280, 550)),
    ('T', (340, 550)), ('Y', (400, 550)), ('U', (460, 550)), ('I', (520, 550)),
    ('O', (580, 550)), ('P', (640, 550)), ('A', (113, 620)), ('S', (173, 620)),
    ('D', (233, 620)), ('F', (293, 620)), ('G', (353, 620)), ('H', (413, 620)),
    ('J', (473, 620)), ('K', (533, 620)), ('L', (593, 620)), ('Z', (128, 690)),
    ('X', (188, 690)), ('C', (248, 690)), ('V', (308, 690)), ('B', (368, 690)),
    ('N', (428, 690)), ('M', (488, 690)), ('Space', (550, 690)), ('Enter', (650, 690)),
    ('Backspace', (750, 690))
]
keys = [Keyboard(text, 50, 50, pos, 6) for text, pos in key_positions]

# Initialize submit button (used in chat interface)
submit_button = Button('Submit', 100, 100, (100, 250), 6)

# Initialize speech-to-text button for chat input
speech_button = Button('Speak', 150, 50, (950, 700), 6)

# Title for chat window
white = (255, 255, 255)
chat_window_title = font.render('Chat Window', True, white)
chat_window_title_rect = chat_window_title.get_rect(center=(500, 50))

# Variable to hold user input and conversation history for chat
user_input_string = ""
conversation_history = []

# Input box dimensions for chat
input_box = pygame.Rect(100, 150, 800, 40)  # x, y, width, height

# Sign-Up Page
# Variables for sign-up information
signup_username = ""
signup_password = ""

def sign_up_page():  # function for the sign up page
    global signup_username, signup_password  # variables that are continually updated based on user input
    active_field = "username"  # Tracks which variable is active ("username" or "password")
    sign_up_active = True
    error_message = ""  # empty string that will be appended to if username or password has not been entered or is too short

    while sign_up_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))  # blank screen for sign up window to show the transition to another window

        # Display Sign Up title on the window
        signup_title = font.render("Sign Up", True, white)
        signup_title_rect = signup_title.get_rect(center=(600, 50))
        screen.blit(signup_title, signup_title_rect)

        # Displays username and interactive text box onto window
        username_label = font.render("Username:", True, white)
        screen.blit(username_label, (100, 150))
        username_box = pygame.Rect(250, 150, 300, 40)
        pygame.draw.rect(screen, white, username_box, 2)
        username_text = font.render(signup_username, True, white)
        screen.blit(username_text, (username_box.x + 5, username_box.y + 5))

        # Displays password and interactive text box onto window
        password_label = font.render("Password:", True, white)
        screen.blit(password_label, (100, 250))
        password_box = pygame.Rect(250, 250, 300, 40)
        pygame.draw.rect(screen, white, password_box, 2)
        # Mask password characters with asterisks
        password_text = font.render(signup_password, True, white)
        screen.blit(password_text, (password_box.x + 5, password_box.y + 5))

        if error_message:  # this is nested if statement for error message
            error_surface = font.render(error_message, True, (255, 0, 0))  # displays the error message onto window
            screen.blit(error_surface, (250, 320))

        # Highlight the active field with a green border
        if active_field == "username":
            pygame.draw.rect(screen, (0, 255, 0), username_box, 2)
        else:
            pygame.draw.rect(screen, (0, 255, 0), password_box, 2)

        for key in keys:
            key_output = key.design_of_keyboard_keys()
            if key_output:  # programming all the keys to work for the username and password
                if key_output == 'Backspace':
                    if active_field == "username":  # variable that keeps track of either username or password
                        signup_username = signup_username[:-1]
                    else:
                        signup_password = signup_password[:-1]
                    error_message = ""  # variable stays empty when expression is valid
                elif key_output == 'Enter':
                    if active_field == "username":
                        # Switch active field to password if username is not empty and meets length requirement
                        if signup_username != "":
                            if len(signup_username) != 8:
                                error_message = "Username must be 8 characters."
                            else:
                                active_field = "password"
                                error_message = ""
                        else:
                            error_message = "Please enter a username"
                    else:
                        # If password field is active, complete sign-up if password is provided and meets length requirement
                        if signup_password != "":
                            if len(signup_password) != 8:
                                error_message = "Password must be 8 characters."
                            else:
                                sign_up_active = False  # If password field is active, sign up is complete
                                error_message = ""
                        else:
                            error_message = "Please enter a password"
                elif key_output == 'Space':
                    if active_field == "username":
                        signup_username += " "
                    else:
                        signup_password += " "
                    error_message = ""
                else:
                    if active_field == "username":
                        signup_username += key_output
                    else:
                        signup_password += key_output
                    error_message = ""

        pygame.display.update()
        clock.tick(10)

#  Login Page
# Variables for login information
login_username = ""
login_password = ""

def login_page():  # function for the login page
    global login_username, login_password  # variables that store user login input
    active_field = "username"  # Tracks which variable is active ("username" or "password")
    login_active = True
    error_message = ""  # empty string that will be appended to if username or password is invalid
    num_of_attempts = 0  # stores the current number of attempts that the user currently has

    while login_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))  # blank screen for login window

        # Display Login title on the window
        login_title = font.render("Login", True, white)
        login_title_rect = login_title.get_rect(center=(600, 50))
        screen.blit(login_title, login_title_rect)

        # Displays username and interactive text box onto window
        username_label = font.render("Username:", True, white)
        screen.blit(username_label, (100, 150))
        username_box = pygame.Rect(250, 150, 300, 40)
        pygame.draw.rect(screen, white, username_box, 2)
        username_text = font.render(login_username, True, white)
        screen.blit(username_text, (username_box.x + 5, username_box.y + 5))

        # Displays password and interactive text box onto window
        password_label = font.render("Password:", True, white)
        screen.blit(password_label, (100, 250))
        password_box = pygame.Rect(250, 250, 300, 40)
        pygame.draw.rect(screen, white, password_box, 2)
        password_text = font.render(login_password, True, white)
        screen.blit(password_text, (password_box.x + 5, password_box.y + 5))

        if error_message:  # nested if statement for error message
            error_surface = font.render(error_message, True, (255, 0, 0))  # displays the error message onto window
            screen.blit(error_surface, (250, 320))

        # Highlight the active field with a green border
        if active_field == "username":
            pygame.draw.rect(screen, (0, 255, 0), username_box, 2)
        else:
            pygame.draw.rect(screen, (0, 255, 0), password_box, 2)

        for key in keys:
            key_output = key.design_of_keyboard_keys()
            if key_output:  # programming all the keys to work for the login username and password
                if key_output == 'Backspace':
                    if active_field == "username":
                        login_username = login_username[:-1]
                    else:
                        login_password = login_password[:-1]
                    error_message = ""
                elif key_output == 'Enter':
                    if active_field == "username":
                        # Switch active field to password if username is not empty
                        if login_username != "":
                            active_field = "password"
                            error_message = ""
                        else:
                            error_message = "Please enter a username"
                    else:
                        # If password field is active, attempt to log in
                        if login_password != "":
                            # Validate login credentials against the sign-up credentials
                            if login_username == signup_username and login_password == signup_password:
                                login_active = False
                                error_message = ""
                            else:
                                num_of_attempts += 1
                                # Switch back to the field that is incorrect
                                if login_username != signup_username:
                                    active_field = "username"
                                    error_message = f"Invalid username {3 - num_of_attempts} attempts remaining."
                                elif login_password != signup_password:
                                    active_field = "password"
                                    error_message = f"Invalid password {3 - num_of_attempts} attempts remaining."
                                if num_of_attempts == 3:  # if num_of_attempts equals 3 then program gets closed
                                    pygame.display.update()
                                    pygame.time.wait(5000)
                                    pygame.quit()
                                    sys.exit()
                        else:
                            error_message = "Please enter a password"
                elif key_output == 'Space':
                    if active_field == "username":
                        login_username += " "
                    else:
                        login_password += " "
                    error_message = ""
                else:
                    if active_field == "username":
                        login_username += key_output
                    else:
                        login_password += key_output
                    error_message = ""

        pygame.display.update()
        clock.tick(10)

# Speech-to-Text Function
def record_audio(duration=5, fs=16000):  # function record_audio initialised with the time duration of the sample interval and the sample rate
    p = pyaudio.PyAudio()  # initialises pyaudio where creates an object for new pyaudio method allowing all packages to be utilised
    try:  # test the new method just created
        stream = p.open(format=pyaudio.paInt16,  # makes sure that the data captured is 16-bit for each sample
                        channels=1,  # audio is captured on 1 channel
                        rate=fs,  # sampling rate or frequency/how many samples per second
                        input=True,  # represent only audio input not output
                        frames_per_buffer=1024)  # number of audio frames per second or processed at one time
    except Exception as e:  # error handling for the microphone
        print("Error opening microphone: ", e)  # error message if microphone doesn't work
        p.terminate()  # if error present, then the pyaudio method/object is closed
        return None, None  # returns no output due to error

    data_file = []  # empty list that would be appended to store new audio data collected from the pyaudio method/object
    print("Recording audio for", duration, "seconds")  # outputs to user that recording has started
    try:
        for _ in range(0, int(fs / 1024 * duration)):  # how many frame would be measured per second for the time duration
            data = stream.read(1024, exception_on_overflow=False)  # reads the 1024 frames of audio data
            data_file.append(data)
    except Exception as e:
        print("Error:", e)  # print any errors
    print("Recording complete.")
    stream.stop_stream()  # stop the pyaudio method
    stream.close()
    p.terminate()
    audio_data = b"".join(data_file)
    return audio_data, fs

def speech_to_text():  # another function is initialised to convert the recorded audio into text
    audio_data, fs = record_audio(duration=5)  # calls the record_audio function with the corresponding parameter of duration of 5 seconds
    if audio_data is None:  # error handling
        return "Microphone error: Unable to record audio."
    r = sr.Recognizer()  # creates instance of recognizer package from SpeechRecognizer library
    # The sample width for paInt16 is 2 bytes.
    audio = sr.AudioData(audio_data, fs, 2)  # formed for processing
    try:
        text = r.recognize_google(audio)  # calling Google API
    except sr.UnknownValueError:
        text = "Could not understand audio."  # catch unclear audio to from user printing error

    return text

# Conversation History
def conversation_history_page():#function used to initialise the conversation history window that will be linked with the chat window
    history_active = True#used to control the running loop of the conversation history
    back_button = Button('Back', 100, 50, (1050, 50), 6)#creates back button for conversation history
    while history_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))  # Clear the screen for the conversation history window

        # Title for the conversation history page
        history_title = font.render("Conversation History", True, white)
        history_title_rect = history_title.get_rect(center=(600, 30))
        screen.blit(history_title, history_title_rect)

        # Display all conversation history inputs and responses
        y_offset = 80  # Starting y-position for the first line
        for line in conversation_history:#iterates for every line outputted by the chat window
            history_line = font.render(line, True, white)#displays on surface of conversation history window
            screen.blit(history_line, (50, y_offset))
            y_offset += 30  # Move down for each subsequent line



        #Draws back button onto the conversation history window
        back_output = back_button.design_of_buttons()
        if back_output == "Back":#Boolean expression to represent that if pressed then will take user back to chat window
            history_active = False  #Returns to chat window

        pygame.display.update()
        clock.tick(10)

# Chat Window

history_button = Button(' Conversation History', 200, 50, (950, 50), 6)
quit_button = Button('Quit', 100, 50, (50, 50), 6)#Creates quit button

# Run the sign-up and login pages before entering the chat window.
sign_up_page()
login_page()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((0, 0, 0))  # Clear the screen
    screen.blit(chat_window_title, chat_window_title_rect)

    # Check if the speech-to-text button is pressed
    speech_output = speech_button.design_of_buttons()
    if speech_output == "Speak":
        spoken_text = speech_to_text()
        user_input_string += spoken_text  # Append speech to chat input

    for key in keys:
        key_output = key.design_of_keyboard_keys()
        if key_output:  # If a key is pressed
            if key_output == 'Space':
                user_input_string += ' '  # Add space to the input
            elif key_output == 'Enter':
                if user_input_string.lower() == "exit":
                    pygame.quit()
                    sys.exit()
                conversation_history.append(f"You: {user_input_string}")

                # Chatbot processing user input through chatbot class
                user_input = Chatbot.tokenize(user_input_string)
                X = Chatbot.bag_of_words(user_input, tokens)
                X = X.reshape(1, X.shape[0])
                X = torch.from_numpy(X).to(device)

                output = chatbot_model(X)
                _, predicted = torch.max(output, dim=1)
                tokens_test_data_labels1 = tokens_test_data_labels[predicted.item()]
                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]

                if prob.item() > 0.75:
                    response = ""
                    for test_data in company_file_test_data['test_data']:
                        if tokens_test_data_labels1 == test_data["type_of_test_data"]:
                            response = random.choice(test_data['responses'])
                    conversation_history.append(f"{bot_name}: {response}")
                else:
                    conversation_history.append(f"{bot_name}: I don't understand, please type your input again.")

                user_input_string = ""  # Clear input string after processing

            elif key_output == 'Backspace':
                user_input_string = user_input_string[:-1]  # Remove last character
            else:
                user_input_string += key_output  # Append the pressed key to the input string

    # Draw the input box for chat
    pygame.draw.rect(screen, (255, 255, 255), input_box, 2)  # Input box outline
    input_surface = font.render(user_input_string, True, white)
    screen.blit(input_surface, (input_box.x + 5, input_box.y + 5))  # Show input text

    # Display conversation history (showing last 10 messages)
    line_separation = 200
    for line in conversation_history[-10:]:
        history_surface = font.render(line, True, white)
        screen.blit(history_surface, (100, line_separation))
        line_separation += 30  # Move down for next line

    #Draws Conversation History button
    history_output = history_button.design_of_buttons()
    if history_output == " Conversation History":#Compares if button was pressed
        conversation_history_page()#Runs Conversation History window

    quit_output = quit_button.design_of_buttons()#Draws quit button
    if quit_output == "Quit":#Boolean expression to compare if button was pressed
        pygame.quit()
        sys.exit()
    pygame.display.update()
    clock.tick(10)
