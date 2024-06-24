import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def preprocess_data(knowledge_base):
    questions = [preprocess_text(item['question']) for item in knowledge_base['questions']]
    answers = [item['answer'] for item in knowledge_base['questions']]
    return questions, answers

def create_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer

def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'w') as file:
        file.write(tokenizer.to_json())

def save_data(data, file_path):
    np.save(file_path, data)

if __name__ == '__main__':
    knowledge_base = load_knowledge_base('knowledge_base.json')
    questions, answers = preprocess_data(knowledge_base)

    tokenizer = create_tokenizer(questions + answers)
    questions_seq = tokenizer.texts_to_sequences(questions)
    answers_seq = tokenizer.texts_to_sequences(answers)

    max_question_length = max(len(seq) for seq in questions_seq)
    max_answer_length = max(len(seq) for seq in answers_seq)

    questions_seq = pad_sequences(questions_seq, maxlen=max_question_length, padding='post')
    answers_seq = pad_sequences(answers_seq, maxlen=max_answer_length, padding='post')

    save_tokenizer(tokenizer, 'tokenizer.json')
    save_data(questions_seq, 'questions_seq.npy')
    save_data(answers_seq, 'answers_seq.npy')
    np.save('max_lengths.npy', [max_question_length, max_answer_length])
