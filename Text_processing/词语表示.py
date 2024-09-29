import pandas as pd
import gensim



import gensim
from gensim.corpora import Dictionary

words=pd.read_csv('df2.csv',encoding='utf-8')
words_list = [words['word'].tolist()]
dct = Dictionary(words_list)
print(dct)

# 将文档转换为词袋表示
corpus = [dct.doc2bow(doc) for doc in documents]
