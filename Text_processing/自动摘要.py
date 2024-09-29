import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, Dense

# 读取诗句数据
df = pd.read_csv('chinese_poems.csv', encoding='utf-8')
poems = df['poem']

# 分词
def tokenize(text):
    return ' '.join(jieba.lcut(text))

poems = poems.apply(tokenize)

# 准备数据
X_train, X_test = train_test_split(poems, test_size=0.2, random_state=42)


