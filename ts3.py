import nltk
import numpy as np
import networkx as nx
import tkinter as tk
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def read_article(text):
    sentences = sent_tokenize(text)
    return sentences

def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(sentence)
    return [word for word in words if word.lower() not in stop_words]

def sentence_similarity(sent1, sent2):
    sent1 = remove_stopwords(sent1)
    sent2 = remove_stopwords(sent2)
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for word in sent1:
        vector1[all_words.index(word)] += 1
    
    for word in sent2:
        vector2[all_words.index(word)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    return similarity_matrix

def generate_summary(text, num_sentences=5):
    sentences = read_article(text)
    similarity_matrix = build_similarity_matrix(sentences)
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([s for score, s in ranked_sentences[:num_sentences]])
    return summary

def summarize_text():
    input_text = text_input.get("1.0", "end-1c")
    num_sentences = int(num_sentences_entry.get())
    summary = generate_summary(input_text, num_sentences)
    summary_output.config(state="normal")
    summary_output.delete("1.0", "end")
    summary_output.insert("1.0", summary)
    summary_output.config(state="disabled")

# Create the GUI
root = tk.Tk()
root.title("Text Summarizer")

# Text Input
text_label = tk.Label(root, text="Input Text:")
text_label.pack()

text_input = tk.Text(root, height=10, width=50)
text_input.pack()

# Number of Sentences Input
num_sentences_label = tk.Label(root, text="Number of Sentences:")
num_sentences_label.pack()

num_sentences_entry = tk.Entry(root)
num_sentences_entry.pack()

# Summarize Button
summarize_button = tk.Button(root, text="Summarize", command=summarize_text)
summarize_button.pack()

# Summary Output
summary_output = tk.Text(root, height=10, width=50, state="disabled")
summary_output.pack()

root.mainloop()
