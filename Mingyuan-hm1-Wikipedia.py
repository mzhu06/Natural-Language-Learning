import urllib
import nltk
from bs4 import BeautifulSoup
from urllib import request
from nltk import FreqDist
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'HW_WikipediaDiscussions.txt')

localURL = "file:///"+filename
openhtml = request.urlopen(localURL)
html = openhtml.read().decode('utf8')
raw = BeautifulSoup(html, features="lxml").get_text()
tokens = nltk.word_tokenize(raw)
#lower text
lower_text = [i.lower() for i in tokens]

#alphabetic
import re
pattern = re.compile('^[^a-z]+$')
nonAlphaMatch = pattern.match('**')
if nonAlphaMatch: print('match non-alphabetical')
def alpha_filter(w): # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False
alpha_text = [ w for w in lower_text if not alpha_filter(w)]

#remove stopwords, because we need to find out the true reason for the discussion.
stopwords = nltk.corpus.stopwords.words('english')
stopped_text = [m for m in alpha_text if m not in stopwords]

#Top 50 Frequency Distribution
fdist=FreqDist(stopped_text)
fdistkeys=list(fdist.keys())

#top 50 frequency table after lower, alpha & stopped
print("Top 50 words in Wikipedia Discussion:")
topkeys=fdist.most_common(50)
for p in topkeys:
    print(p)

#Biagram Frequenct Distribution (start with only lower filtered data)
from nltk.collocations import *
biagram_measures = nltk.collocations.BigramAssocMeasures()

finder = BigramCollocationFinder.from_words(lower_text)
scored = finder.score_ngrams(biagram_measures.raw_freq)
#apply alpha_filter
finder.apply_word_filter(alpha_filter)
scored = finder.score_ngrams(biagram_measures.raw_freq)
print("Top 50 biagram in Wikipedia Discussion:")
for biscore in scored[:50]:
    print(biscore)

#Mutual Information
finder2 = BigramCollocationFinder.from_words(stopped_text)
scored2 = finder2.score_ngrams(biagram_measures.pmi)
finder2.apply_freq_filter(5)
print('Top 50 PMI bigrams by their Mutual Information scores (min freq 5) in Wikipedia Discussion: ')
for bscore in scored2[:50]:
    print(bscore)