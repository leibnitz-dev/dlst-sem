from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Information retrieval is the process of obtaining relevant information",
    "TF IDF stands for term frequency inverse document frequency",
    "Python is widely used for data science and machine learning",
    "Machine learning improves information retrieval systems"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

query = input()
query_vector = vectorizer.transform([query])

similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
for i, score in enumerate(similarity_scores[0]):
    print(f"Document {i+1} similarity score: {score:.4f}")

most_relevant_index = similarity_scores.argmax()
print("Most relevant document:")
print(documents[most_relevant_index])
