import requests
from bs4 import BeautifulSoup
import csv

# 请求URL并获取响应内容
with open('comments.csv', 'a', newline='', encoding='utf-8') as file:
    # 打开CSV文件进行写操作
    writer = csv.writer(file)
    # 写入CSV头
    for i in range(0,501,20):
        url = f'https://book.douban.com/subject/1136881/reviews?start={i}'
        headers = {
            "User-Agent": "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; Acoo Browser; GTB5; Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1) ; InfoPath.1; .NET CLR 3.5.30729; .NET CLR 3.0.30618)"
        }
        response = requests.get(url=url, headers=headers)
        data = response.text

        # 解析HTML数据
        soup = BeautifulSoup(data, 'lxml')
        comment_items = soup.select('div.review-short')

        # 打开CSV文件进行写操作

        for item in comment_items:
            user = item.select_one('a.name')
            time = item.select_one('span.main-meta')
            content = item.select_one('div.short-content').get_text().strip()

            writer.writerow([user,time,content])

        print("Data has been written to reviews.csv")
