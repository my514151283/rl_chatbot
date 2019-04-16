from collections import defaultdict
from model_config import PAD_WORD, NOT_DEFINED_WORD, START_WORD, END_WORD
from model_config import VOCABULARY_COUNT, DATASET_LENGTH, VOCABULARY_SIZE


#########################################  载入post和response数据集  ###################################################
posts_string = []  # post列表
with open("./data/stc_weibo_train_post", "r", encoding="utf-8") as posts:
    line_count = 0
    for line in posts:
        # line.strip().split(" ")
        # ['竟然', '下雪', '了', '，', '不', '喜欢', '冬天', '，', '天气', '何时', '才', '变暖', '啊', '。']
        line = line[: -1].split(" ")
        # [['竟然', '下雪', '了', '，', '不', '喜欢', '冬天', '，', '天气', '何时', '才', '变暖', '啊', '。'],
        #  ['人生', '如', '戏', '，', '全', '靠', '演技', '。']]
        posts_string.append(line)
        line_count += 1
        # 读取DATASET_LENGTH条post
        if line_count % DATASET_LENGTH == 0:
            print("posts读取完成，读取数量：", line_count)
            break

responses_string = []  # response列表
with open("./data/stc_weibo_train_response", "r", encoding="utf-8") as responses:
    line_count = 0
    for line in responses:
        # ['记得', '雪莱', '的', '诗句', '吗', '？', '＂', '冬天', '来', '了', '，', '春天', '还', '会', '远', '吗', '？', '＂']
        line = line[: -1].split(" ")
        responses_string.append(line)
        line_count += 1
        if line_count % DATASET_LENGTH == 0:
            print("responses读取完成，读取数量：", line_count)
            break
########################################################################################################################

##########################################  载入预训练的词向量  ########################################################
vocabulary = {}  # 载入腾讯的词汇表
with open("./data/Tencent_AILab_ChineseEmbedding.txt", "r", encoding="utf-8") as embeds:
    line_count = 0
    for line in embeds:
        # 跳过第一行数据 8824330 200
        if line_count == 0:
            line_count += 1
            continue
        # 建立词向量字典{'word':[embed]}
        line = line[: -1].split(" ")
        word = line[0]
        embed = line[1:]
        # {'</s>': ['0.002001', '0.002210',...],'',:[],...}
        vocabulary[word] = embed
        line_count += 1
        if line_count % VOCABULARY_SIZE == 0:  # 载入词向量个数
            print("读取词向量完成，读取", line_count, "行")
            break
########################################################################################################################

#######################################  统计数据集中每个单词的词频，降序排序  #########################################
'''
在python中如果访问字典中不存在的键，会引发KeyError异常
defaultdict(int)使用int类型来初始化一个dict
对于不存在的键，其默认值为0
'''
word_frequency = defaultdict(int)  # post和response用到的词汇和词频
for post in posts_string:
    for word in post:
        word_frequency[word] += 1
for response in responses_string:
    for word in response:
        word_frequency[word] += 1
# print(type(word_frequency))  # defaultdict {'我': 5, '的': 3, '你': 10}

# 根据word_frequency.items()[1]降序排列
word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
# print(type(word_frequency))  # list [('你', 10), ('我', 5), ('的', 3)]

used_vocabulary = []  # 4个特殊单词 + post和response用到的词汇(按照其词频从高到低排序)
for word, frequency in word_frequency:
    if frequency >= 5:
        used_vocabulary.append(word)
# ['_PAD', '_NDW', '你', '我', '的']
used_vocabulary = [PAD_WORD] + [NOT_DEFINED_WORD] + [START_WORD] + [END_WORD] + used_vocabulary
print("posts和responses使用的词汇数量：", len(used_vocabulary))

with open("./data/word_frequency.txt", "w", encoding='utf-8') as file:
    for word, frequency in word_frequency:
        if frequency >= 5:
            file.write('{0}:{1}\n'.format(word, frequency))
########################################################################################################################

#######################################  截取需要个数的词汇表和词向量  #################################################
with open("./data/vocabulary.txt", "w", encoding='utf-8') as file:
    word_count = 0
    for word in used_vocabulary:
        # file.write(word + "\n")
        file.writelines(word + "\n")
        word_count += 1
        if word_count % VOCABULARY_COUNT == 0:
            print("选取词汇表大小：", VOCABULARY_COUNT)
            break

# 不存在词向量的词汇200维的0填充
with open("./data/embed.txt", "w", encoding='utf-8') as file:
    word_count = 0
    for word in used_vocabulary:
        if word in vocabulary.keys():
            file.writelines(" ".join(vocabulary[word]) + "\n")
            word_count += 1
        else:
            file.writelines(" ".join(['0'] * 200) + "\n")
            word_count += 1
        if word_count % VOCABULARY_COUNT == 0:
            break
########################################################################################################################

