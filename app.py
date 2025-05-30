import spacy
import nltk
from nltk.corpus import wordnet as wn
import streamlit as st
from transformers import pipeline
from spacy.tokens import Doc
import spacy_streamlit
import re

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model and transformers pipeline
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    st.error("Please install the spaCy model: python -m spacy download en_core_web_lg")
    exit(1)

# Load a transformers model for contextual sense disambiguation (BERT-based)
try:
    sense_disambiguator = pipeline("text-classification", model="bert-base-uncased", tokenizer="bert-base-uncased")
except Exception as e:
    st.warning(f"Could not load transformers model: {e}. Falling back to WordNet senses.")

def get_word_sense(word, pos, sentence, use_transformers=False):
    """Retrieve WordNet sense or use transformers for contextual disambiguation."""
    if use_transformers and 'sense_disambiguator' in globals():
        # Simplified contextual sense prediction (mock implementation)
        synsets = wn.synsets(word, pos=pos)
        if not synsets:
            return None
        # Use BERT to score synset definitions
        scores = []
        for syn in synsets:
            definition = syn.definition()
            result = sense_disambiguator(f"{word} in context: {sentence}. Definition: {definition}")
            scores.append((syn.name(), result[0]['score']))
        return max(scores, key=lambda x: x[1])[0] if scores else None
    else:
        synsets = wn.synsets(word, pos=pos)
        return synsets[0].name() if synsets else None

def extract_semantic_frame(sentence):
    """Extract a semantic frame from a sentence with enhanced parsing and NER."""
    try:
        doc = nlp(sentence)
        frame = {
            "action": None,
            "agent": None,
            "theme": None,
            "location": None,
            "attributes": {},
            "entities": []
        }
        
        # Extract entities (NER)
        for ent in doc.ents:
            frame["entities"].append({
                "text": ent.text,
                "label": ent.label_,
                "sense": get_word_sense(ent.text.lower(), wn.NOUN, sentence, use_transformers=True)
            })
        
        # Parse dependency tree
        for token in doc:
            # Main verb (action)
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj"):
                frame["action"] = {
                    "lemma": token.lemma_,
                    "sense": get_word_sense(token.lemma_, wn.VERB, sentence)
                }
            
            # Subject (agent)
            if token.dep_ in ("nsubj", "nsubjpass"):
                frame["agent"] = {
                    "text": token.text,
                    "sense": get_word_sense(token.lemma_, wn.NOUN, sentence, use_transformers=True)
                }
            
            # Object (theme)
            if token.dep_ in ("dobj", "pobj", "attr"):
                frame["theme"] = {
                    "text": token.text,
                    "sense": get_word_sense(token.lemma_, wn.NOUN, sentence, use_transformers=True)
                }
            
            # Location (prepositional phrases)
            if token.dep_ == "prep" and token.head.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ in ("pobj", "npadvmod"):
                        frame["location"] = {
                            "text": child.text,
                            "sense": get_word_sense(child.lemma_, wn.NOUN, sentence)
                        }
            
            # Attributes (adjectives, adverbs)
            if token.pos_ in ("ADJ", "ADV"):
                frame["attributes"][token.text] = {
                    "lemma": token.lemma_,
                    "sense": get_word_sense(token.lemma_, wn.ADJ if token.pos_ == "ADJ" else wn.ADV, sentence)
                }
        
        return frame, doc
    except Exception as e:
        st.error(f"Error processing sentence: {e}")
        return None, None

def display_frame(frame):
    """Format the semantic frame for display."""
    if not frame:
        return "Error: Could not generate semantic frame."
    
    output = "Semantic Frame:\n"
    if frame["action"]:
        output += f"  Action: {frame['action']['lemma']} ({frame['action']['sense']})\n"
    if frame["agent"]:
        output += f"  Agent: {frame['agent']['text']} ({frame['agent']['sense']})\n"
    if frame["theme"]:
        output += f"  Theme: {frame['theme']['text']} ({frame['theme']['sense']})\n"
    if frame["location"]:
        output += f"  Location: {frame['location']['text']} ({frame['location']['sense']})\n"
    if frame["attributes"]:
        output += "  Attributes:\n"
        for attr, details in frame["attributes"].items():
            output += f"    {attr}: {details['lemma']} ({details['sense']})\n"
    if frame["entities"]:
        output += "  Named Entities:\n"
        for ent in frame["entities"]:
            output += f"    {ent['text']}: {ent['label']} ({ent['sense']})\n"
    return output

# Streamlit UI
def main():
    st.title("Enhanced Semantic Frame Builder")
    st.write("Enter a sentence to generate a semantic frame and visualize its dependency tree.")
    
    # Sentence input
    sentence = st.text_input(
        "Enter a sentence:",
        "The quick brown fox jumps over the lazy dog in London."
    )
    
    # Options
    use_transformers = st.checkbox("Use contextual sense disambiguation (BERT)", value=False)
    
    if st.button("Generate Semantic Frame"):
        if sentence and re.match(r'^[A-Za-z0-9\s.,!?]+$', sentence):
            frame, doc = extract_semantic_frame(sentence)
            if frame and doc:
                # Display semantic frame
                st.subheader("Generated Semantic Frame")
                st.text_area("Frame Output", display_frame(frame), height=300)
                
                # Visualize dependency tree
                st.subheader("Dependency Tree Visualization")
                spacy_streamlit.visualize_parser(doc)
            else:
                st.error("Failed to process the sentence. Please try again.")
        else:
            st.warning("Please enter a valid sentence (letters, numbers, spaces, and basic punctuation only).")

if __name__ == "__main__":
    # Example usage in console
    example_sentence = "The quick brown fox jumps over the lazy dog in London."
    frame, _ = extract_semantic_frame(example_sentence)
    print(display_frame(frame))
    
    # Run Streamlit UI
    main()