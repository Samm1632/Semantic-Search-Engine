import streamlit as st
from sentence_transformers import SentenceTransformer
import json
import hnswlib
import enchant
title_list=[]
def correct_spelling(text):
    spellchecker = enchant.Dict("en_US")
    words = text.split()
    corrected_words = []
    for word in words:
        if spellchecker.check(word):
            corrected_words.append(word)
        else:
            suggestions = spellchecker.suggest(word)
            if suggestions:
                corrected_words.append(suggestions[0])
            else:
                corrected_words.append(word)
    return " ".join(corrected_words)
model= SentenceTransformer("all-mpnet-base-v2")
loaded_index = hnswlib.Index(space='cosine', dim=768)  
loaded_index.load_index('C:/Users/SHYAM/Projects/Mastek/Final_code/semantic_search/sematic_search.bin')
with open('C:/Users/SHYAM/Projects/Mastek/Final_code/semantic_search/data_title.json', 'r') as json_file:
    data_title = json.load(json_file)
st.text_input("What can I find you?",key="query")
if "query" in st.session_state:
    search_term=st.session_state.query
    corrected_term=correct_spelling(search_term)
    query_embed=model.encode(corrected_term)
    labels, Distances = loaded_index.knn_query(query_embed, k=20)
    for label in labels[0]:
        title=data_title[label]
        title_list.append(title)    
    st.write(title_list)