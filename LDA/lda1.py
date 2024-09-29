from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

categories = ['sci.med', 'sci.space', 'comp.graphics']
newsgroups_data = fetch_20newsgroups(categories=categories)

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(newsgroups_data.data)

n_topics = 10
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(X)

# 获取每个主题下的top_words个高频词
top_words = 10
feature_names = vectorizer.get_feature_names_out()

fig, axes = plt.subplots(2, n_topics // 2, figsize=(15, 10), sharex=True)
axes = axes.flatten()
for topic_idx, topic in enumerate(lda_model.components_):
    top_features_ind = topic.argsort()[:-top_words - 1:-1]
    top_features = [feature_names[i] for i in top_features_ind]
    weights = topic[top_features_ind]

    ax = axes[topic_idx]
    ax.barh(top_features, weights, height=0.7)
    ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 10})
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=10)
    for i in 'top right left'.split():
        ax.spines[i].set_visible(False)

plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.5)
plt.show()