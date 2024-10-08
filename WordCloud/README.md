一、功能概述

此项目主要实现了对中文文本的处理和分析，包括读取文本文件、统计文本字数、进行不同方式的分词、去除停用词、统计词频以及生成词云图（注释部分）。

二、代码详解

读取文本

指定文本文件路径file_path，使用open函数以只读模式和指定编码读取文本内容，并存储在chinese_text变量中。

统计文本的字数并打印输出。

分词操作

分别使用不同的方法进行分词：

jieba.cut返回一个生成器，可迭代得到分词结果，将结果转换为列表并写入文件output1.txt。
jieba.lcut直接返回列表形式的分词结果，写入文件output2.txt。
jieba.lcut_for_search进行更适合搜索的分词，结果写入文件output3.txt。
加载自定义的搜狗词库文件bleach.txt后，使用jieba.lcut进行分词，结果写入文件output4.txt。

去除停用词

读取停用词库文件停词库.txt到stop_word的DataFrame中，对中文文本进行分词后，过滤掉在停用词列表中的词语，得到处理后的分词结果列表words_list5，并写入文件output5.txt。

统计词频

遍历处理后的分词结果列表，统计每个词的出现频率，存储在字典word_freq中。然后对词频字典进行排序，得到按词频从高到低排列的字典sorted_word_freq并打印输出。

生成词云图（注释部分）

读取指定的图片作为词云的形状掩码mask。
创建词云对象cloud_obj，指定字体路径、掩码、背景颜色、停用词列表和最大词数，并从词频字典生成词云。
使用掩码图像的颜色重新给词云上色。
使用matplotlib显示生成的词云图。

三、应用场景

这段代码可用于文本分析、自然语言处理、信息检索等领域，具体应用如下：

文本挖掘：通过分析文本中的高频词汇，了解文本的主题和关键信息。

情感分析：结合特定的情感词典，可以分析文本的情感倾向。

内容推荐：根据文本的关键词，为用户推荐相关的内容。
