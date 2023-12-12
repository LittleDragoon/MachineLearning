import string
import pdb
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import re
porter_stemmer = PorterStemmer()

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    punctuationfree = punctuationfree.replace('\n', ' ').lower()
    final_text = re.sub(r'\d+', '', punctuationfree)
    return final_text

def tokenization(text):
    tk = WhitespaceTokenizer()
    return tk.tokenize(text)

def stemming(text):
  stem_text = [porter_stemmer.stem(word) for word in text]
  return stem_text

#> 'The striped bat be hang on their foot for best'
def preprocess_text(text):
    text_without_punctuation = remove_punctuation(text)
    tokenized_text = tokenization(text_without_punctuation)
    stemmed_text = stemming(tokenized_text)
    return ' '.join(stemmed_text)





