import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 读取数据
df = pd.read_csv('/Users/gushuai/Desktop/bigdata/上的各种课/大三下/社交网络与文本分析/大作业/comments.csv', encoding='utf-8')
df = df.iloc[:, 2]
df_cleaned = df.dropna()

# 读取停词表
with open(r"/Users/gushuai/Desktop/文件夹/学习文件/大三上/组织行为学/停词库.txt") as f:
    stoplist = f.read().split()
stoplist += ['\n', ',', ' ']

# 对每个评论进行分词并去除停词
def tokenize(text):
    tokens = jieba.lcut(text)
    return [token for token in tokens if token not in stoplist]

df_cleaned = df_cleaned.apply(str)
tokenized_docs = df_cleaned.apply(tokenize)

# 将分词后的文本合并为字符串
corpus = [" ".join(tokens) for tokens in tokenized_docs]

# 计算TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# 获取TF-IDF得分最高的前10个词
top_n = 10
feature_names = vectorizer.get_feature_names_out()
top_keywords = []
for i in range(X.shape[0]):
    indices = X[i].toarray().flatten().argsort()[-top_n:][::-1]
    keywords = [feature_names[idx] for idx in indices]
    top_keywords.append(keywords)

print(top_keywords)
import jieba.analyse

# 使用 TextRank 提取关键词
def extract_keywords_textrank(text, top_k=10):
    keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
    return keywords

textrank_keywords = df_cleaned.apply(extract_keywords_textrank)

print(textrank_keywords)
import gensim
from gensim import corpora

# 创建词典和语料库
texts = [tokenize(text) for text in df_cleaned]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 训练LDA模型
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# 显示每个主题的前10个词
topics = lda_model.print_topics(num_words=6)
for topic in topics:
    print(topic)
