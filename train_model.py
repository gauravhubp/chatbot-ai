import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional # type: ignore
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore

def load_data(file_path):
    return np.load(file_path)

def load_tokenizer(file_path):
    with open(file_path, 'r') as file:
        tokenizer = tokenizer_from_json(file.read())
    return tokenizer

if __name__ == '__main__':
    questions_seq = load_data('questions_seq.npy')
    answers_seq = load_data('answers_seq.npy')
    max_lengths = np.load('max_lengths.npy')
    max_question_length, max_answer_length = max_lengths
    tokenizer = load_tokenizer('tokenizer.json')
    vocab_size = len(tokenizer.word_index) + 1

    print("questions_seq shape:", questions_seq.shape)
    print("answers_seq shape:", answers_seq.shape)
    print("vocab_size:", vocab_size)

    # Modify answers_seq to be a 1D array of integer labels
    answers_seq = answers_seq[:, 0]
    print("Modified answers_seq shape:", answers_seq.shape)

    # Ensure correct data types
    questions_seq = questions_seq.astype(np.int32)
    answers_seq = answers_seq.astype(np.int32)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_question_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(questions_seq, answers_seq, epochs=150, validation_split=0.2)

    model.save('chatbot_model.h5')