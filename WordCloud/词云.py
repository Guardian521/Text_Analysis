import pandas as pd
import wordcloud
import wordcloud as wc
import matplotlib.pyplot as plt
import jieba
file_path = 'chinese_text.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    chinese_text = file.read()

#统计字数
char_count = len(chinese_text)
print('文字数量：', char_count)

import jieba

'''
#分词
# 1
res1 = jieba.cut(chinese_text)
words_list1 = [word for word in res1]
with open("output1.txt", "w", encoding="utf-8") as file:
    file.write('/'.join(words_list1))

#2
res2 = jieba.lcut(chinese_text)
words_list2 = [word for word in res2]
with open("output2.txt", "w", encoding="utf-8") as file:
    file.write('/'.join(words_list2))

# 3
res3 = jieba.lcut_for_search(chinese_text)
words_list3 = [word for word in res3]
with open("output3.txt", "w", encoding="utf-8") as file:
    file.write('/'.join(words_list3))

# 使用搜狗词库分词
jieba.load_userdict('bleach.txt')
res4 = jieba.lcut(chinese_text)
words_list4 = [word for word in res4]
with open("output4.txt", "w", encoding="utf-8") as file:
    file.write('/'.join(words_list4))
'''
#停词
stop_word=pd.read_csv('停词库.txt', names=['w'], sep='\t', encoding='utf-8')
words_list5 = [word for word in jieba.cut(chinese_text) if word not in set(stop_word['w'].tolist())]

with open("output5.txt", "w", encoding="utf-8") as file:
    file.write('/'.join(words_list5))

#词频
word_freq = {}
for word in words_list5:
    if len(word) > 1:  # 过滤单个字符的词语
        word_freq[word] = word_freq.get(word, 0) + 1

sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

print(sorted_word_freq)
'''
#词云图
import numpy as np
from PIL import Image

mask_img = '截屏2024-04-09 17.14.58.png'
mask = np.array(Image.open(mask_img))

# 创建词云对象
font_path = "/Users/gushuai/Library/Fonts/方正硬笔行书简体.ttf"
cloud_obj = wordcloud.WordCloud(font_path=font_path,
                                mask=mask,
                                background_color='white',
                                stopwords=stop_word['w'].tolist(),
                                max_words=3000).generate_from_frequencies(sorted_word_freq)

# 生成词云图
image_colors = wordcloud.ImageColorGenerator(np.array(mask))
cloud_obj.recolor(color_func=image_colors)

plt.imshow(cloud_obj, interpolation='bilinear')
plt.axis("off")
plt.show()'''