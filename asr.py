
"!pip install librosa numpy tensorflow scikit-learn"

"!git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git"

import os
dataset_path = "/content/free-spoken-digit-dataset/recordings"
files = os.listdir(dataset_path)
print("Total files:", len(files))
print("Sample files:", files[:5])

import IPython.display as ipd
sample_file = dataset_path + "/0_jackson_0.wav"
ipd.Audio(sample_file)

import librosa
import librosa.display
import matplotlib.pyplot as plt
y, sr = librosa.load(sample_file, sr=8000)

plt.figure()
librosa.display.waveshow(y, sr=sr)
plt.title(f'waveform for {dataset_path + "/0_jackson_0.wav" }')
plt.show()

import numpy as np
import librosa

x = []; y = []

def extract_features(file_path):
     acoustic_signal , sr = librosa.load(file_path, sr= 8000)
     mfcc = librosa.feature.mfcc(y=acoustic_signal, sr=sr, n_mfcc =13)
     return mfcc.T

for f in os.listdir(dataset_path):
        if f.endswith(".wav"):
           label = int(f.split("_")[0]) # audio filename consists of both the digit & the path details.
           path = os.path.join(dataset_path, f)

           features = extract_features(path)
           x.append(features)
           y.append(label)

#x[0].shape

# padding mfcc features
max_length = max([feature.shape[0] for feature in x])

def pad(feature_to_pad):
    if feature_to_pad.shape[0] < max_length:
       return np.pad(feature_to_pad, ((0, max_length - feature_to_pad.shape[0]), (0,0)))
    return feature_to_pad[:max_length]

x = np.array([pad(feature) for feature in x])
y = np.array(y)

# add channel dimension for CNN

x.shape; x = x[..., np.newaxis]; print(x.shape) # To make the audio mfcc fetaures compatible with cnn.

"""<pre>
  Always end with four dimensions for CNN (sample, time, features, channels); can be visualized as a spectrogram  image of sound over time.
</pre>
"""

print('x[0] shape:', end=""); print(x[0].shape);print('x shape:', end=""); print(x.shape)

import tensorflow
from tensorflow.keras import models, layers

model = models.Sequential([
                          layers.Conv2D(32, (3,3), activation = 'relu', input_shape = x.shape[1:]),
                          layers.MaxPooling2D(2,2),

                          layers.Conv2D(64, (3,3), activation = 'relu'),
                          layers.MaxPooling2D(2,2),

                          layers.Flatten(),
                          layers.Dense(64, activation = 'relu'),
                          layers.Dense(10, activation = 'softmax')
                          ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.evaluate(x,y)

# model.fit(x, y, epochs=20, batch_size=32, validation_split=0.2)
from sklearn.model_selection import train_test_split

# to avoid overfitting issue
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model.fit(x_train, y_train, epochs=20)
model.evaluate(x_test, y_test)

def predict(file):
    feat = extract_features(file)
    feat = pad(feat)
    feat = feat[np.newaxis, ..., np.newaxis]

    pred = model.predict(feat)
    return np.argmax(pred)
test = ["/content/free-spoken-digit-dataset/recordings/1_nicolas_3.wav", "/content/free-spoken-digit-dataset/recordings/0_lucas_12.wav"]
print("Predicted digit:", predict(test[0]))

model.save("digit_asr_model_train_test_split.keras")

"""<pre>
load the model
</pre>
"""

from tensorflow.keras.models import load_model
model = load_model("digit_asr_model.keras")

def predict(file):
    import librosa
    import numpy as np

    signal, sr = librosa.load(file, sr=8000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T

    # pad
    if mfcc.shape[0] < max_length:
        mfcc = np.pad(mfcc, ((0, max_length - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:max_length]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(mfcc)
    return np.argmax(pred)

print(predict("/content/free-spoken-digit-dataset/recordings/1_nicolas_3.wav"))

