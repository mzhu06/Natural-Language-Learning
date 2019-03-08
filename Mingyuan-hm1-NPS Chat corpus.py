import nltk
from nltk.corpus import nps_chat
nps_chat.fileids()
chatroom1 = nps_chat.fileids()[1]
chatroom1 = nps_chat.posts(chatroom1)
chatwords_list = []
for w in chatroom1:
    chatwords_list.append(' '.join(w))
chatwords = ' '.join(chatwords_list)
#tokenization
chat_token = nltk.word_tokenize(chatwords)
print(chat_token)
#lower & alpha
lower_chat = [w.lower() for w in chat_token]
#alpha_chat = [w for w in lower_chat if w.isalpha()]
#stop words
stopwords = nltk.corpus.stopwords.words('english')
stopped_chat = [m for m in lower_chat if m not in stopwords]
#Frequency Table
from nltk import FreqDist
fdist=FreqDist(lower_chat)
print("Top 50 words in NPS-chat corpus [1]:")
topkeys=fdist.most_common(50)
for p in topkeys:
    print(p)
#Bigram Frequency
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(lower_chat)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print("Top 50 biagram in NPS-chat corpus [1]:")
for bscore in scored[:50]:
    print (bscore)

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

finder1 = BigramCollocationFinder.from_words(stopped_chat)
finder1.apply_freq_filter(5)
scored1 = finder.score_ngrams(bigram_measures.pmi)
print("Top 50 PMI biagram in NPS-chat corpus [1] with min freq 5:")
for j in scored1[:50]:
     print(j)
