import spacy
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample review
review = "I love my new Sony headphones! The sound quality is amazing. Beats can't compare."

# Named Entity Recognition
doc = nlp(review)
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Rule-based sentiment analysis using TextBlob
blob = TextBlob(review)
sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative"
print(f"\nSentiment: {sentiment}")