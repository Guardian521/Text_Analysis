#julei
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 读取数据
df = pd.read_csv('/Users/gushuai/Desktop/bigdata/上的各种课/大三下/社交网络与文本分析/大作业/comments.csv', encoding='utf-8')
df = df.iloc[:, 2]
df_cleaned = df.dropna()

# 读取停词库
with open(r"/Users/gushuai/Desktop/文件夹/学习文件/大三上/组织行为学/停词库.txt") as f:
    stoplist = f.read()
stoplist = stoplist.split() + ['\n', ',', ' ']

# 对每条评论进行分词并过滤停词
def preprocess(text):
    item_words = jieba.lcut(str(text))
    return ' '.join([word for word in item_words if word not in stoplist])

df_cleaned = df_cleaned.apply(preprocess)

# 向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_cleaned)

# 使用KMeans进行聚类
num_clusters = 5  # 选择适当的聚类数
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# 获取每个评论的聚类标签
labels = kmeans.labels_

# 将聚类结果加入原数据中
df_result = pd.DataFrame({'comment': df_cleaned, 'cluster': labels})

# 可视化聚类结果（使用PCA降维到2D以便可视化）
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(X.toarray())

colors = ["r", "b", "c", "y", "m"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(x_axis, y_axis, c=[colors[d] for d in labels])

plt.title('KMeans Clustering of Comments')
plt.show()

# 输出每个聚类中的一些评论
for i in range(num_clusters):
    print(f"\nCluster {i+1}:")
    cluster_comments = df_result[df_result['cluster'] == i]['comment']
    print(cluster_comments.head(5))  # 输出前5条评论



