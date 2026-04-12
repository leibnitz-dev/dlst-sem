!pip install gliner
from gliner import GLiNER

# Try the base model which is highly stable and public
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

text = "I met Mathandam and he said me about assignments and I got 1000 rupees"
# Expanded labels for a "big" program
labels = [
    "person",   # People
    "GPE",      # Cities/Countries
    "org",      # Companies/Institutions
    "date",     # Dates
    "money",    # Currency
    "product",  # Software/Hardware/Goods
    "loc",      # Rivers/Mountains/Regions
    "event",    # Elections/Meetings/Games
    "language", # Python/English/Tamil
    "quantity"  # Weights/Measurements/Distance
]

entities = model.predict_entities(text, labels)

for entity in entities:
    print(f"{entity['text']} => {entity['label']}")


pip install spacy
python -m spacy download en_core_web_sm

import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Elon Musk founded SpaceX in the United States."

# Process text
doc = nlp(text)

# Print entities
for ent in doc.ents:
    print(ent.text, ent.label_)
