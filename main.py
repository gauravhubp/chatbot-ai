import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import get_close_matches
from typing import Optional

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_knowledge_base(file_path: str, data: dict) -> None:
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def find_best_match(user_question: str, questions: list[str]) -> Optional[str]:
    matches = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_answer_for_question(question: str, knowledge_base: dict) -> Optional[str]:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]
    return None

def load_tokenizer(file_path):
    with open(file_path, 'r') as file:
        tokenizer = tokenizer_from_json(file.read())
    return tokenizer

def chat_bot():
    knowledge_base = load_knowledge_base('knowledge_base.json')
    model = load_model('chatbot_model.h5')
    tokenizer = load_tokenizer('tokenizer.json')
    max_lengths = np.load('max_lengths.npy')
    max_question_length = max_lengths[0]

    while True:
        user_input = input('You: ')
        
        if user_input.lower() == 'quit':
            break
        
        preprocessed_input = preprocess_text(user_input)
        input_seq = tokenizer.texts_to_sequences([preprocessed_input])
        input_seq = pad_sequences(input_seq, maxlen=max_question_length, padding='post')
        
        pred = model.predict(input_seq)
        pred_index = np.argmax(pred, axis=1)[0]
        response = tokenizer.index_word.get(pred_index, '')

        best_match = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])
        
        if best_match:
            answer = get_answer_for_question(best_match, knowledge_base)
            print(f'Bot: {answer}')
        elif response:
            print(f'Bot: {response}')
        else:
            print('Bot: I don\'t know the answer. Can you teach me?')
            new_answer = input('Type the answer or "skip" to skip: ')
            
            if new_answer.lower() != 'skip':
                knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                print('Bot: Thank you! I learned a new response!')

if __name__ == '__main__':
    chat_bot()