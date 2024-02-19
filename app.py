from transformers import BertTokenizerFast
from transformers import AutoModelForTokenClassification
from transformers import pipeline
import streamlit as st
from spacy_streamlit import visualize_ner
import spacy
from spacy import displacy

st.set_page_config(layout="wide")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("multilingual_ner")
colors = {
    
    'AGE': '#8CB9BD',
    'DATE': '#A3C9AA',
    'GENDER': '#BBE2EC',
    'JOB': '#474F7A',
    'LOCATION': '#6962AD',
    'NAME': '#D24545',
    'ORGANIZATION': '#E7BCDE',
    'PATIENT_ID': '#9EB8D9',
    'SYMPTOM_AND_DISEASE': '#DCBFFF',
    'TRANSPORTATION': '#D6D46D',
    'O': '#FFF5E0',
}
def merge_consecutive_entities(entities):
    merged_entities = []
    if entities:
        current_ent = entities[0]
        for ent in entities[1:]:
            if ent.label_ == current_ent.label_:
                current_ent.end = ent.end
            else:
                merged_entities.append(current_ent)
                current_ent = ent
        merged_entities.append(current_ent)
    return merged_entities

def visualize_ner(input_text):
    
    nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)
    data = nlp(input_text) 
    nlp = spacy.blank("en")
    doc = nlp(input_text)

    # Populate the NER annotations
    entities = [(item['start'], item['end'], item['entity'].split('-')[1]) for item in data]
    custom_entities = []
    for span_start, span_end, label in entities:
        ent = doc.char_span(span_start, span_end, label=label)
        if ent is None:
            continue
        custom_entities.append(ent)

    merged_entities = merge_consecutive_entities(custom_entities)
    doc.ents = merged_entities  
    options = {"ents": list(set(label['entity'].split('-')[1] for label in data)), "colors": colors}
    html = displacy.render(doc, style='ent',options=options, page=True)
    st.write(html, unsafe_allow_html=True)

def main():
    st.title("NAMED ENTITY RECOGNITION VỚI TRANSFORMER (PROJECT I)")
    st.sidebar.title("Input Text")
    
    # Input text
    input_text = st.sidebar.text_area("Enter text:", value="", placeholder="Type here...",height=200)
    if st.sidebar.button("Visualize"):
        visualize_ner(input_text)

if __name__ == "__main__":
    main()
