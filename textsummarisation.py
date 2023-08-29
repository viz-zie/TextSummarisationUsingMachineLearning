import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

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

# Example text
input_text = """
Natural language processing (NLP) is a subfield of artificial intelligence that focuses
 on the interaction between computers and humans through natural language. The ultimate 
 goal of NLP is to read, decipher, understand, and make sense of human language in a way 
 that is both valuable and meaningful. NLP helps computers understand human language to 
 facilitate interactions and enable effective communication. It involves various challenges 
 such as language understanding, language generation, machine translation, sentiment analysis, 
 and more.
"""

cnt1=0
ar=input_text.split()
for i in range (0,len(ar)):
    cnt1+=1
    print(ar[i])


print(cnt1)

summary = generate_summary(input_text)
print("Summary:")
print(summary)


cnt2=0
arr=summary.split()
for i in range (0,len(arr)):
    cnt2+=1
    print(arr[i])


print(cnt2)
