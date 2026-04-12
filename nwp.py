import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = load_dataset("wikitext", "wikitext-2-v1", split='train')
type(dataset)

raw_text = [line for line in dataset['text'] if len(line) > 100][:1000]


print('sample dataset demonstrting few lines:\n',raw_text[:2])

corpus = " ".join(raw_text).lower()
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts([corpus])
total_words = len(tokenizer.word_index) + 1

# Convert the corpus into a list of integers
token_list = tokenizer.texts_to_sequences([corpus])[0]

X = []
y = []

for i in range(2, len(token_list)):
    trigram = token_list[i-2:i+1]
    X.append(trigram[:2])  # first 2 words as input
    y.append(trigram[2])   # third word as target

X = np.array(X)
y = np.array(y)

print(f"Total training sequences: {len(X)}")
print(f"Sample Input (X): {X[0]} -> Output (y): {y[0]}")

# ----------------------------
# 2. Define LSTM model
# ----------------------------
model = tf.keras.Sequential([
    # Embedding layer: vocab_size × 100-dimensional vectors
    tf.keras.layers.Embedding(
        input_dim=vocab_size,   # size of vocabulary
        output_dim=100,         # embedding dimension
        input_shape=(2,)        # sequence length = 2
    ),

    # LSTM layer: 150 units
    tf.keras.layers.LSTM(150),

    # Dense output: predict next word
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Buildthe model explicitly (optional but ensures summary shows parameters)
model.build(input_shape=(None, 2))

# Show summary
model.summary()

model.fit(X, y, epochs=10, batch_size=32)

import pickle

# 1. Save the trained LSTM Model
model.save('trigram_lstm_model.h5')
print("Model saved to trigram_lstm_model.h5")

# 2. Save the Tokenizer (The most important part for NLP!)
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved to tokenizer.pkl")

loss, accuracy = model.evaluate(X, y)
perplexity = np.exp(loss)
print("Perplexity:", perplexity)


############ Using Trained Model #####################
import tensorflow as tf
import pickle
import numpy as np

# Load trained model
model = tf.keras.models.load_model('trigram_lstm_model.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_next_word(seed_text):
    # 1. Clean and tokenize the input string
    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]

    # 2. Ensure we only tak the last 2 words (trigram logic)
    token_list = token_list[-2:]

    # 3. Reshape for the model (Batch size of 1, Sequence length of 2)
    token_list = np.array([token_list])

    # 4. Get probabilities for the entire vocabulary
    predictions = model.predict(token_list, verbose=0)

    # 5. Get the index of the word with the highest probability
    predicted_index = np.argmax(predictions, axis=-1)[0]


    # 6. Map the integer back to the actual word
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    return output_word
