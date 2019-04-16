from model_config import DATASET_LENGTH, PAD_WORD, START_WORD, END_WORD, BATCH_SIZE, VOCABULARY_COUNT
from model_config import NUM_LAYERS, NUM_UNITS, LEARNING_RATE, MAX_GRADIENT_NORM, MAX_LENGTH
from model_config import TRAIN
import numpy as np
from model import model
import tensorflow as tf
import copy
import sys
import jieba

def read_dataset():
    '''
    读取训练集Post和Response
    :return:
    post_string = [['竟然', '下雪', '了', '，', '不', '喜欢', '冬天', '，', '天气', '何时', '才', '变暖', '啊', '。'],[],...]
    response_string = [['记得', '雪莱', '的', '诗句', '吗', '？', '＂', '冬天', '来', '了', '，', '春天', '还', '会', '远', '吗', '？', '＂'],[],...]
    '''
    post_string = []
    with open("./data/stc_weibo_train_post", "r", encoding="utf-8") as posts:
        line_count = 0
        for line in posts:
            line = line[: -1].split(" ")
            post_string.append(line)
            line_count += 1
            if line_count % DATASET_LENGTH == 0:
                print("posts读取完成，读取数量：", line_count)
                break
    #print("第200条post：", post_string[199])

    response_string = []
    with open("./data/stc_weibo_train_response", "r", encoding="utf-8") as responses:
        line_count = 0
        for line in responses:
            line = line[: -1].split(" ")
            response_string.append(line)
            line_count += 1
            if line_count % DATASET_LENGTH == 0:
                print("responses读取完成，读取数量：", line_count)
                break
    #print("第200条response：", response_string[199])
    return post_string, response_string

def read_embed():
    '''
    读取词嵌入向量
    :return:
    embed = [[ 0.        0.        ...  0.        ]
             ...
             [ 0.274581 -0.274623  ... -0.202317  ]]
    '''
    embed = []
    with open("./data/embed.txt", "r", encoding="utf-8") as embeds:
        for line in embeds:
            line = line[: -1].split(" ")
            # str转换为float
            line = [float(item) for item in line]
            embed.append(line)
    # [[0.0, 0.0, ..., 0.0],...,[0.274581, -0.274623, ..., -0.202317]]
    for item in embed:
        if len(item) != 200:
            print("词向量维度错误")
    print("词向量数量：", len(embed))
    # 将list转换为array
    embed = np.array(embed, dtype=np.float32)
    # print("第100个词向量：", embed[99])
    return embed

def read_vocabulary():
    '''
    读取词汇
    :return:  vocabulary = ['_PAD', '_NDW', '_GO', '_END', '，', ...]
    '''
    vocabulary = []
    with open("./data/vocabulary.txt", "r", encoding="utf-8") as vocabularies:
        for line in vocabularies:
            line = line[: -1]
            vocabulary.append(line)
    print("词汇表大小：", len(vocabulary))
    # print("第100个词汇：", vocabulary[99])
    return vocabulary

def get_batch_data(post, response):
    '''
    得到一批数据，返回补齐后的数据
    :param post: [['竟然', '下雪', '了', '，', '不', '喜欢', '冬天', '，', '天气', '何时', '才', '变暖', '啊', '。'],[],...]
    :param response: [['记得', '雪莱', '的', '诗句', '吗', '？', '＂', '冬天', '来', '了', '，', '春天', '还', '会', '远', '吗', '？', '＂'],[],...]
    :return:
    batch_data = {"post": post,  # [[raw_post + '_PAD'], ...]
                  "response": response,  # [['_GO' + raw_response + '_PAD'], ...]
                  "label": label,  # [[raw_response + '_END' + '_PAD'], ...]
                  "post_len": post_len,  # [5 7 ... ]每个Post序列的长度
                  "response_len": response_len  # [6 4  ...]每个response序列的长度}
    response ['_GO','我','吃','过','了','_PAD','PAD']
    label ['我','吃','过','了','_END', '_PAD','PAD']
    ??????????????
    ??????????????为什么label不用np.array
    '''
    post_len = [len(item) for item in post]
    # +1 是为了在response前加[START_WORD]
    response_len = [len(item)+1 for item in response]

    encoder_len = max(post_len)
    # print("post的最大长度：", encoder_len)
    decoder_len = max(response_len)
    # print("response的最大长度：", decoder_len)

    # 补齐post的长度
    for index in range(len(post)):
        post[index] = post[index] + [PAD_WORD] * (encoder_len-len(post[index]))

    # 复制response到label中
    label = copy.deepcopy(response)

    # 补齐response的长度
    for index in range(len(response)):
        response[index] = [START_WORD] + response[index] + [PAD_WORD] * (decoder_len-1-len(response[index]))
    # 补齐label的长度
    for index in range(len(label)):
        label[index] = label[index] + [END_WORD] + [PAD_WORD] * (decoder_len-1-len(label[index]))

    post = np.array(post)
    response = np.array(response)
    post_len = np.array(post_len, dtype=np.int32)
    response_len = np.array(response_len, dtype=np.int32)

    batch_data = {"post": post,
                  "response": response,
                  "label": label,
                  "post_len": post_len,
                  "response_len": response_len}
    return batch_data

if __name__ == '__main__':
    # 读取选定数据集长度的post和response，返回列表[DATASET_LENGTH * post_len],[DATASET_LENGTH * response_len]
    post, response = read_dataset()
    # 读取词嵌入，返回np数组，[VOCABULARY_COUNT * 200]
    embed = read_embed()
    # 读取词汇表，返回列表[1 * VOCABULARY_COUNT]
    vocabulary = read_vocabulary()
    # 生成长度为len(vocabulary)的列表 [0, 1, 2, ..., len(vocabulary)-1]
    vocabulary_index = list(range(len(vocabulary)))
    num_data_set = len(post)  # DATASET_LENGTH

    seq2seq = model(embed=embed,
                    vocabulary=vocabulary,
                    vocabulary_count=VOCABULARY_COUNT,
                    num_layers=NUM_LAYERS,
                    num_units=NUM_UNITS,
                    learning_rate=LEARNING_RATE,
                    max_gradient_norm=MAX_GRADIENT_NORM,
                    max_len=MAX_LENGTH)

    with tf.Session() as sess:
        # checkpoint文件会记录保存信息，通过它可以定位最新保存的模型
        if tf.train.get_checkpoint_state("./train/"):
            print("从记录中恢复模型参数！")
            # tf.train.latest_checkpoint（）自动获取最后一次保存的模型
            seq2seq.saver.restore(sess, tf.train.latest_checkpoint("./train/"))
            '''
            打印 参数名字(name):对应维度(shape)
            embed:0: (85326, 200)
            encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0: (456, 1024)
            encoder/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0: (1024,)
            encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0: (512, 1024)
            encoder/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0: (1024,)
            decoder/decoder_rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0: (456, 1024)
            decoder/decoder_rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0: (1024,)
            decoder/decoder_rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0: (512, 1024)
            decoder/decoder_rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0: (1024,)
            decoder/decoder_rnn/projection_layer/weights:0: (256, 90000)
            decoder/decoder_rnn/projection_layer/biases:0: (90000,)
            global_step:0: ()
            '''
            seq2seq.print_parameters()
        else:
            print("重新创建模型参数！")
            # 初始化模型参数
            tf.global_variables_initializer().run()
            # 给表赋初值 table.insert(key_list, value_list)
            sess.run(seq2seq.string_to_id.insert(tf.constant(vocabulary, dtype=tf.string), tf.constant(vocabulary_index, dtype=tf.int64)))
            sess.run(seq2seq.id_to_string.insert(tf.constant(vocabulary_index, dtype=tf.int64), tf.constant(vocabulary, dtype=tf.string)))

        # 训练
        if TRAIN:
            while(True):
                start = 0
                end = BATCH_SIZE
                total_loss = 0.0
                while(start < num_data_set):
                    if end > num_data_set:
                        end = num_data_set
                    # 获得一批格式化过的数据
                    # data = {"post":[[]], "response":[[]], "label":[[]], "post_len":[], "response_len":[]}
                    data = get_batch_data(post[start: end], response[start: end])

                    # 生成feed_dict = {input1:[], input2:[]}，给使用placeholder创建出来的tensor赋值
                    feed_data = {seq2seq.post_string: data["post"],
                                 seq2seq.response_string: data["response"],
                                 seq2seq.label_string: data["label"],
                                 seq2seq.post_len: data["post_len"],
                                 seq2seq.response_len: data["response_len"]}

                    batch_size = end - start
                    # 梯度更新，计算损失
                    _, loss, avg_loss = sess.run([seq2seq.update, seq2seq.loss, seq2seq.avg_loss], feed_dict=feed_data)
                    total_loss += loss
                    print("start=", start, "end=", end-1, "每条数据的平均损失：", loss/batch_size, "，每个单词平均损失：", avg_loss)
                    start = end
                    end += BATCH_SIZE
                print("结束一轮的训练，记录模型参数。每条数据平均损失：", total_loss/num_data_set)
                # global_step表示当前是第几步, -(global_step).meta
                # global_step += DATASET_LENGTH/BATCH_SIZE
                seq2seq.saver.save(sess, "./train/", global_step=seq2seq.global_step)
        # 测试
        else:
            # data = get_batch_data(post[:15], response[:15])
            sys.stdout.write("> ")
            sys.stdout.flush()
            input_seq = input()

            while input_seq:
                input_seq = input_seq.strip()
                seg_list = list(jieba.cut(input_seq))
                user_input = [seg_list]
                user_input_len = [len(user_input[0])]
                user_input = np.array(user_input)
                user_input_len = np.array(user_input_len, dtype=np.int32)

                feed_data = {seq2seq.post_string: user_input,
                             seq2seq.post_len: user_input_len,}
                inference_string = sess.run(seq2seq.inference_string, feed_dict=feed_data)
                # [['你', '好', '啊', '_END', '_NDW', '_NDW'],[],..]
                inference_string = [[str(word, encoding="utf-8") for word in response] for response in inference_string.tolist()]
                for response in inference_string:
                    try:
                        sentense = response[: response.index(END_WORD)]
                    except Exception as e:
                        sentense = response
                    sentense = " ".join(sentense)
                    # ['你', '好', '啊']
                    print(sentense)
                sys.stdout.write("> ")
                sys.stdout.flush()
                input_seq = input()



