import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# 1. Load Dataset
# ----------------------------
dataset = load_dataset("wikitext", "wikitext-2-v1", split='train')

raw_text = [line for line in dataset['text'] if len(line) > 100][:1000]

print("Sample text:\n", raw_text[:2])

# Join and lowercase
corpus = " ".join(raw_text).lower()

# ----------------------------
# 2. Character Tokenization
# ----------------------------
chars = sorted(list(set(corpus)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

vocab_size = len(chars)

print("Total unique characters:", vocab_size)

# Convert entire text into integers
text_as_int = np.array([char2idx[c] for c in corpus])

# ----------------------------
# 3. Create Sequences
# ----------------------------
seq_length = 40   # number of input characters
X = []
y = []

for i in range(len(text_as_int) - seq_length):
    X.append(text_as_int[i:i+seq_length])
    y.append(text_as_int[i+seq_length])

X = np.array(X)
y = np.array(y)

print("Total sequences:", len(X))
print("Example input:", X[0])
print("Example target:", y[0])

# ----------------------------
# 4. Build Model
# ----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=seq_length),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# 5. Train Model
# ----------------------------
model.fit(X, y, epochs=2, batch_size=64)

# ----------------------------
# 6. Save Model
# ----------------------------
model.save("char_lstm_model.h5")

import pickle

with open("char_mapping.pkl", "wb") as f:
    pickle.dump((char2idx, idx2char), f)

print("Model and mappings saved!")

# ----------------------------
# 7. Evaluate
# ----------------------------
loss, accuracy = model.evaluate(X, y)
perplexity = np.exp(loss)

print("Perplexity:", perplexity)

# ----------------------------
# 8. Text Generation
# ----------------------------
def generate_text(seed_text, next_chars=200, temperature=0.7):
    input_seq = [char2idx[c] for c in seed_text.lower() if c in char2idx]

    for _ in range(next_chars):
        padded = pad_sequences([input_seq], maxlen=seq_length, truncating='pre')

        preds = model.predict(padded, verbose=0)[0]

        # Temperature sampling
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        next_index = np.random.choice(len(preds), p=preds)
        next_char = idx2char[next_index]

        input_seq.append(next_index)
        seed_text += next_char

    return seed_text


# ----------------------------
# 9. Test Generation
# ----------------------------
print(generate_text("the meaning of life is"))
