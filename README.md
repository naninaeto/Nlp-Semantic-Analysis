#  Semantic Frame Builder

This project is a Streamlit web application that extracts and visualizes semantic frames from input sentences. It uses **spaCy** for syntactic parsing and named entity recognition, **NLTK WordNet** for word sense disambiguation, and optionally leverages a **transformers BERT-based model** for contextual sense disambiguation.

---

## Features

- Extract semantic frames from sentences with:
  - Action (main verb)
  - Agent (subject)
  - Theme (object)
  - Location (prepositional phrase)
  - Attributes (adjectives and adverbs)
  - Named Entities with labels and senses
- Contextual word sense disambiguation using WordNet and optionally BERT-based transformers.
- Dependency tree visualization using spaCy-Streamlit.
- Simple and interactive web UI for input and visualization.

--------------------------------------------------------------------------|
|streamlit — for the web app interface                                    |
                                                                          |
|spacy — core NLP library                                                 |

|en_core_web_lg — the large English spaCy model (installed via URL)       |
                                                                          |
|nltk — for WordNet and other lexical data                                |
                                                                          |
|transformers — HuggingFace transformers for BERT disambiguation          |
                                                                          |
|torch — required by transformers as the backend                          |
                                                                          |
|spacy-streamlit — for dependency tree visualization                      |
--------------------------------------------------------------------------|

## Installation

1. Clone this repository:

   ```bash
   https://github.com/naninaeto/Nlp-Semantic-Analysis.git
   cd semantic-frame-builder
   


python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run app.py


<!-- 
"John gave Mary a book."

"The teacher assigned homework to the students."

"Sarah sent Tom a message."
# clear example 
# Mary reads a book.

# The cat chased the mouse.

# Alice is writing a letter.

# Tom played the guitar.

# The teacher gave homework.


# These sentences are designed to include:

# Clear subjects, verbs, and objects

# Different verb tenses and structures

# Common semantic roles (e.g., agent, action, object)
 -->
