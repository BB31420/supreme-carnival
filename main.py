# main.py
from chatbot import Chatbot

if __name__ == "__main__":
    document_directory = "./data_3d_printing"
    chatbot = Chatbot(document_directory)
    chatbot.chat_loop()