import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imread
import numpy as np
#分词和可视化
df=pd.read_csv('/Users/gushuai/Desktop/bigdata/上的各种课/大三下/社交网络与文本分析/大作业/comments.csv',encoding='utf-8')
df=df.iloc[:,2]
df_cleaned = df.dropna()
with open(r"/Users/gushuai/Desktop/文件夹/学习文件/大三上/组织行为学/停词库.txt") as f:
    stoplist=f.read()
stoplist=stoplist.split()+['\n',',',' ']
# 使用jieba对“互动感知”列中的每一个项进行分词，并过滤掉停词
words = []
for item in df:
    item_words = jieba.lcut(str(item))
    words.extend([word for word in item_words if word not in stoplist])




df = pd.DataFrame(words, columns = ['word'])
df.to_csv('df2.csv',encoding='utf-8')
df.head()
result = df.groupby(['word']).size()
# print(result)
freqlist = result.sort_values(ascending=False) # 降序
a=freqlist[:20]
print(a)


from imageio import imread
wc = WordCloud(font_path="/Users/gushuai/Library/Fonts/方正硬笔行书简体.ttf",  # 这里是字体的路径，如 '/Users/gushuai/Library/Fonts/方正硬笔行书简体.ttf'
               mask=imread('111.png'),
               background_color='white',
               width=800,
               height=600)

# 从词频字典生成词云
wc.generate_from_frequencies(freqlist)

# 显示词云图
plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.savefig(r"/Users/gushuai/Desktop/文件夹/学习文件/大三上/组织行为学/wordcloud2.png", format='png')
plt.show()


