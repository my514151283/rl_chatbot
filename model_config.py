'''
常量（全部大写）参数设置
数据集 400W
词向量 800w+ 200维
'''

PAD_WORD = "_PAD"  # 填充句子的字符
NOT_DEFINED_WORD = "_NDW"  # 词汇表中未出现的单词
START_WORD = "_GO"  # 一句话的起始的单词
END_WORD = "_END"  # 一句话的结束的单词
PAD_WORD_ID = 0
NOT_DEFINED_WORD_ID = 1
START_WORD_ID = 2
END_WORD_ID = 3

VOCABULARY_COUNT = 52604  # 词汇表大小  2500
DATASET_LENGTH = 350000  # 数据集大小  512
VOCABULARY_SIZE = 800000  # 腾讯词向量读取的数量

BATCH_SIZE = 32  # 每批样本的大小
NUM_LAYERS = 2  # encoder和decoder的层数
NUM_UNITS = 256  # encoder和decoder的隐藏状态维度
LEARNING_RATE = 0.0001  # 学习率
MAX_GRADIENT_NORM = 5.0  # 梯度裁剪参数
MAX_LENGTH = 60  # 解码最大长度

TRAIN = True  # True（训练） or False（测试）
