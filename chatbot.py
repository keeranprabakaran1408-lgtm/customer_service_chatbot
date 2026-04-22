import pygame
import sys
from transformers import pipeline

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
FONT_SIZE = 24
BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
BUTTON_COLOR = (100, 200, 100)
BUTTON_HOVER_COLOR = (150, 250, 150)


class Chatbot:
    def __init__(self):
        # Use text-generation instead of conversational
        self.model = pipeline("text-generation", model="microsoft/DialoGPT-medium")
        self.chat_history = []

    def get_response(self, user_input):
        # Generate a response using the model
        response = self.model(user_input, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']

    def add_to_history(self, message):
        self.chat_history.append(message)


class ChatApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Customer Service Chatbot")
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.chatbot = Chatbot()
        self.user_input = ""

    def draw_chat_history(self):
        y_offset = 50
        for message in self.chatbot.chat_history:
            text_surface = self.font.render(message, True, TEXT_COLOR)
            self.screen.blit(text_surface, (20, y_offset))
            y_offset += 30
            if y_offset > HEIGHT - 50:  # Prevent overflow
                break

    def draw_input_box(self):
        pygame.draw.rect(self.screen, (200, 200, 200), (20, 350, 460, 40), 2)
        input_surface = self.font.render(self.user_input, True, TEXT_COLOR)
        self.screen.blit(input_surface, (30, 355))

    def draw_send_button(self):
        button_rect = pygame.Rect(490, 350, 80, 40)
        mouse_pos = pygame.mouse.get_pos()
        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, BUTTON_HOVER_COLOR, button_rect)
        else:
            pygame.draw.rect(self.screen, BUTTON_COLOR, button_rect)

        button_text = self.font.render("Send", True, TEXT_COLOR)
        self.screen.blit(button_text, (button_rect.x + 20, button_rect.y + 5))

    def run(self):
        clock = pygame.time.Clock()

        while True:
            self.screen.fill(BACKGROUND_COLOR)  # Clear the screen with the background color

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKSPACE:
                        self.user_input = self.user_input[:-1]
                    elif event.key == pygame.K_RETURN:
                        if self.user_input.strip():
                            self.chatbot.add_to_history(f"You: {self.user_input}")
                            response = self.chatbot.get_response(self.user_input)
                            self.chatbot.add_to_history(f"Bot: {response}")
                            self.user_input = ""
                    else:
                        self.user_input += event.unicode

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if 490 <= mouse_pos[0] <= 570 and 350 <= mouse_pos[1] <= 390:
                        if self.user_input.strip():
                            self.chatbot.add_to_history(f"You: {self.user_input}")
                            response = self.chatbot.get_response(self.user_input)
                            self.chatbot.add_to_history(f"Bot: {response}")
                            self.user_input = ""

            self.draw_chat_history()
            self.draw_input_box()
            self.draw_send_button()

            pygame.display.flip()  # Update the display
            clock.tick(30)  # Limit the frame rate


if __name__ == "__main__":
    app = ChatApp()
    app.run()