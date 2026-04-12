import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

model = Sequential([
    Embedding(5000, 32),
    SimpleRNN(32, dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

def predict_sentiment(text):
    word_index = imdb.get_word_index()
    words = text.lower().split()
    indices = [word_index.get(w, 2) + 3 for w in words]
    padded = pad_sequences([indices], maxlen=100)

    score = model.predict(padded)[0][0]
    label = "Positive" if score > 0.5 else "Negative"

    return label, f"{score:.4f}"

# Example Usage:
print(predict_sentiment("This movie was fantastic and I loved every minute"))
print(predict_sentiment("The plot was boring and the acting was terrible"))

model.save('imdb_rnn_model.keras')

loaded_model = tf.keras.models.load_model('imdb_rnn_model.keras')
print(predict_sentiment("This movie was a flop but the actor did well"))

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Basic Evaluation (Loss and Accuracy)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc)

# 2. Detailed Class-wise Metrics
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# 3. Visualizing Errors
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
