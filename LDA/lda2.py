from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud

with open('news_corpus.pkl', 'rb') as f:
    documents = pickle.load(f)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)

n_topics = 10
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(X)

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

for topic_idx, topic in enumerate(lda_model.components_):
    wordcloud = WordCloud(width=800, height=400, font_path='/System/Library/Fonts/PingFang.ttc',
                          background_color='white').generate_from_frequencies(dict(zip(feature_names, topic)))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topic {topic_idx + 1} Word Cloud', fontsize=14)
    plt.show()


# 绘制词云
for topic_idx, topic in enumerate(lda_model.components_):
    topic_features = ' '.join([feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]])
    wordcloud = WordCloud(width=800, height=400, font_path='/System/Library/Fonts/PingFang.ttc',
                          background_color='white').generate(' '.join(top_features))

    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'主题 {topic_idx + 1} 词云', fontsize=12)
    plt.show()
