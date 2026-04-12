import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# Load models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"   # better than distilbert
)

# Load index + data
index = faiss.read_index("faiss_index.index")
data = pickle.load(open("passages.pkl", "rb"))


def get_answer(query, top_k=5):
    # Step 1: Encode query
    query_vec = embed_model.encode([query])

    # Step 2: Retrieve top passages
    D, I = index.search(query_vec, top_k)
    contexts = [data.iloc[i]["passage"] for i in I[0] if 0 <= i < len(data)]

    if not contexts:
        return "No relevant context found."

    # Step 3: Re-rank
    pairs = [[query, c] for c in contexts]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, contexts), reverse=True)
    contexts = [c for _, c in ranked]

    # Step 4: Combine top contexts (limit size)
    combined_context = " ".join(contexts[:3])[:1000]

    # Step 5: QA
    result = qa_pipeline({
        "question": query,
        "context": combined_context
    })

    answer = result.get("answer", "").strip()
    score = float(result.get("score", 0.0))

    # Step 6: Validate answer
    if score > 0.4 and len(answer) > 3:
        return answer

    # Fallback
    return contexts[0]

########### Experiment 7: ASR ###############

from tensorflow.keras.models import load_model
m = load_model("/content/digit_asr_model_train_test_split.keras")

def recognize_audio_for_digit(file):
    import librosa
    import numpy as np

    signal , sr = librosa.load(file, sr=8000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T

    max_length =36
    if mfcc.shape[0] < max_length:
         mfcc = np.pad(mfcc, ((0, max_length- mfcc.shape[0]), (0,0)))
    else:
         mfcc = mfcc[:max_length]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    pred = m.predict(mfcc)
    return int(np.argmax(pred))

recognize_audio_for_digit("/content/nine.wav")
