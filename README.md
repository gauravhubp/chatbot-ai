# chatbot-ai

## Project Overview

This project implements an AI-powered chatbot that combines a neural network model with a static knowledge base. The chatbot is designed to answer user queries using either pre-defined responses or generated responses based on its training.

## Features

1. Neural Network Model: Utilizes a deep learning model trained on question-answer pairs to generate responses.
2. Static Knowledge Base: Maintains a JSON-based knowledge base of pre-defined question-answer pairs.
3. Natural Language Processing: Employs NLTK for text preprocessing, including tokenization and stopword removal.
4. Fuzzy Matching: Uses difflib for finding close matches to user queries in the knowledge base.

## Project Structure

- `train_model.py`: Script for training the neural network model.
- `preprocess.py`: Script for preprocessing the training data.
- `main.py`: The main script that runs the chatbot.
- `knowledge_base.json`: JSON file storing the question-answer pairs.
- `chatbot_model.h5`: Saved neural network model.
- `tokenizer.json`: Saved tokenizer for text processing.
- `max_lengths.npy`: Numpy file storing maximum lengths for input sequences.

## Setup and Running

1. Install the required dependencies:

  - pip install tensorflow nltk

2. Download the necessary NLTK data:

   - import nltk
   - nltk.download('punkt')
   - nltk.download('stopwords')

4. Preprocess the chatbot
   - python preprocess.py
   
3.Train the model by running 
   - python train_model.py.

4.Run the chatbot:
   - python main.py

5.Type "quit" to stop the chatbot.  

# Future Improvements

Implement more advanced NLP techniques for better understanding of user queries.
Enhance the neural network model with attention mechanisms or transformer architecture.
Add support for multi-turn conversations and context understanding.
Implement a web interface for easier interaction with the chatbot.
