from collections import defaultdict
import numpy as np
import tensorflow as tf

# posts_string = []
# with open("./data/stc_weibo_train_post", "r", encoding="utf-8") as posts:
#     line_count = 0
#     for idx, line in enumerate(posts):
#         if idx < 30:
#             print(idx)
#             print(line)
#             print(line[: -1])
#             line = line.strip().split(" ")
#             print(line)
#             posts_string.append(line)
#             line_count += 1
#     print(posts_string)

# vocabulary = {}  # 载入腾讯的词汇表
# with open("./data/Tencent_AILab_ChineseEmbedding.txt", "r", encoding="utf-8") as embeds:
#     line_count = 0
#     for line in embeds:
#         # 跳过第一行数据 8824330 200
#         if line_count == 0:
#             line_count += 1
#             continue
#         # 建立词向量字典{'word':'embed'}
#         line = line[: -1].split(" ")
#         word = line[0]
#         embed = line[1:]
#         vocabulary[word] = embed
#         line_count += 1
#         if line_count % 5 == 0:  # 载入词向量个数
#             print("读取词向量完成，读取", line_count, "行")
#             break
#     print(vocabulary)

# word_frequency = {'我':5, '的':3, '你':10}
# # print(word_frequency)
# # print(type(word_frequency))
# word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
# print(type(word_frequency))
# print(word_frequency)
# with open("./data/word_frequency.txt", "w", encoding='utf-8') as file:
#     for word in word_frequency:
#         file.write('{0}:{1}\n'.format(word[0], word[1]))

# used_vocabulary = []  # 4个特殊单词 + post和response用到的词汇(按照其词频从高到低排序)
# for word, frequency in word_frequency:
#     used_vocabulary.append(word)
# used_vocabulary = ["_PAD"] + ["_NDW"] + used_vocabulary
# print(used_vocabulary)

# word_frequency = defaultdict(int)
# print(word_frequency['a'])
# print(word_frequency['b'])
# word_frequency['a'] += 1
# word_frequency['b'] = 2
# word_frequency['c'] = 3
# word_frequency['e'] = 4
# print(word_frequency['a'])
# print(word_frequency.items())
# print(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))
# vocabulary = {'a':'1','b':'2','c':'3','d':'4'}
# wordlist = ['a', 'b', 'c', 'd']
# for word in wordlist:
#     if word in vocabulary.keys():
#         print(" ".join(vocabulary[word]))
# print('#'*100)
# used_vocabulary = ['a', 'b', 'c', 'd', 'e']
# with open("./data/test.txt", "w", encoding='utf-8') as file:
#     file.writelines(used_vocabulary)

# with open("./data/test.txt", "w", encoding='utf-8') as file:
#     for word in used_vocabulary:
#         file.write(word + "\n")


# embed = []
# with open("./data/embed.txt", "r", encoding="utf-8") as embeds:
#     for idx, line in enumerate(embeds):
#         if idx < 30:
#             line = line[: -1].split(" ")
#             print(line)
#             line = [float(item) for item in line]
#             print(line)
#             embed.append(line)
#         else:
#             break
#     print(embed)
# embed = np.array(embed, dtype=np.float32)
# print(embed)

# vocabulary = []
# with open("./data/vocabulary.txt", "r", encoding="utf-8") as vocabularies:
#     for idx, line in enumerate(vocabularies):
#         if idx < 30:
#             line = line[: -1]
#             vocabulary.append(line)
# print(vocabulary)
#
# vocabulary_index = list(range(20))
# print(type(vocabulary_index))
# print(len(vocabulary_index))
# print(vocabulary_index)


# post_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="post_len")
# print(post_len)
# feed = {post_len: [1,2,3]}
# onehot = tf.one_hot([1, 2, 3], 3, axis=0)
# with tf.Session() as sess:
#     print(sess.run(post_len, feed_dict=feed))
#     print(post_len.get_shape())
#     print(sess.run(onehot))


# a = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]
# a = np.asarray(a)
# idx1 = tf.Variable([0, 2, 3, 1], tf.int32)
# idx2 = tf.Variable([[0, 2, 3, 1], [4, 0, 2, 2]], tf.int32)
# out1 = tf.nn.embedding_lookup(a, idx1)
# out2 = tf.nn.embedding_lookup(a, idx2)
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(out1))
#     print(out1)
#     print('==================')
#     print(sess.run(out2))
#     print(out2)
#
# aa = ['a' for _ in range(2)]
# print(aa)


# x = [[[0,1,2],
#        [3,4,5],
#        [6,7,8],
#        [9,0,1]],
#      [[9,8,7],
#       [0,0,0],
#       [6,5,4],
#       [3,2,1]]]

# input_ta = tf.TensorArray(size=2, dtype=tf.float32, clear_after_read=False)
# print(input_ta)
# output = input_ta.unstack(x)
# print()
#
# next_cell_input = output.read(0)
# print(next_cell_input)
#
# next_cell_input_id = tf.ones([8], dtype=tf.int32) * 2
# print(next_cell_input_id)
# with tf.Session() as sess:
#     print (sess.run(next_cell_input_id))


# import tensorflow as tf
# batch_size = 4
# input = tf.random_normal(shape=[3, batch_size, 6], dtype=tf.float32)
# cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)
# init_state = cell.zero_state(batch_size, dtype=tf.float32)
# output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state, time_major=True)
# #time_major如果是True，就表示RNN的steps用第一个维度表示，建议用这个，运行速度快一点。
# #如果是False，那么输入的第二个维度就是steps。
# #如果是True，output的维度是[steps, batch_size, depth]，反之就是[batch_size, max_time, depth]。就是和输入是一样的
# #final_state就是整个LSTM输出的最终的状态，包含c和h。c和h的维度都是[batch_size， n_hidden]
# # batch_size = 4 steps = 3 num_units = 10
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print('#'*100)
#     print(sess.run(output))  # 3*4*10
#     print('#' * 100)
#     print(sess.run(final_state))
#     print('#' * 100)
#     print(sess.run([output,final_state]))

# next_done = tf.zeros([8], dtype=tf.bool)
# with tf.Session() as sess:
#     print(sess.run(next_done))




# # batch_size = 2,decoder_len = 4, vocabulary_count = 3
# A = [[1.0, 2.0],
#      [1.0, 2.0]]
# A = [[[0.0,1.0,2.0],
#        [3.0,4.0,5.0],
#        [6.0,7.0,8.0],
#        [9.0,0.0,1.0]],
#      [[9.0,8.0,7.0],
#       [0.0,0.0,0.0],
#       [6.0,5.0,4.0],
#       [3.0,2.0,1.0]]]
# with tf.Session() as sess:
#         print(sess.run(tf.nn.softmax(A)))  # batch_size * decoder_len * vocabulary_count

# probability = tf.constant([[1.0, 2.0],[3.0, 4.0],[2.0, 0.0]], dtype=tf.float32)
# argmax = tf.argmax(probability, axis=1)
# done = tf.equal(argmax, 1)
# with tf.Session() as sess:
#     print(argmax.eval())
#     print(done.eval())


# sum1 = [[ 1,  2,  3],
#         [ 5,  7,  9],
#         [12, 15, 18]]
# response_len = tf.constant([1,2,3], dtype=tf.int32)
# decoder_len = tf.constant(4, dtype=tf.int32)
# onehot = tf.one_hot(response_len-1, decoder_len)
# mask = tf.cumsum(onehot, axis=1, reverse=True)
# label_mask = tf.reshape(mask, [-1])
# total_size = tf.reduce_sum(label_mask)
# with tf.Session() as sess:
#     print(onehot.eval())
#     print(mask.eval())
#     print(label_mask.eval())
#     print(total_size.eval())

# reA = tf.reshape(A, [-1])
# with tf.Session() as sess:
#     print(sess.run(reA))

# response_len = tf.constant([2,4,5,2,2], dtype=tf.int32)
# decoder_len = tf.constant(5, dtype=tf.int32)
# a = tf.one_hot(response_len-1, decoder_len)
# b = tf.cumsum(a, axis=1, reverse=True)
# label_mask = tf.reshape(b, [-1])
# total_size = tf.reduce_sum(label_mask)
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(b))
#     print(sess.run(label_mask))
#     print(sess.run(total_size))

# inference_string = [['你','好','啊','_END','_NDW','_NDW'],['你','吃','了','吗','_END','_NDW'],['下','次','见','_END','_NDW','_NDW']]
# # for response in inference_string:
# #     print(response)
# inference_string = [[word for word in response] for response in inference_string]
# print(inference_string)
# for response in inference_string:
#     try:
#         sentense = response[: response.index('_END')]
#     except Exception as e:
#         sentense = response
#         sentense = " ".join(sentense)
#     print(sentense)

# import sys
# import jieba
# sys.stdout.write("> ")
# sys.stdout.flush()
# input_seq = input()
# seg_list = list(jieba.cut(input_seq))
# user_input = [seg_list]
# print(user_input)
# # print(len(user_input))
# print([len(user_input[0])])
# # print([len(item) for item in user_input])
# post_len = [len(user_input[0])]
# encoder_len = max(post_len)
# user_input[0] = user_input[0] + ["_PAD"] * (encoder_len - len(user_input[0]))
# print(user_input)

# query = tf.constant([[1,2],[3,4],[5,6]],dtype = tf.float32)
# h = tf.expand_dims(query, 2)
# value = tf.constant([[[1,1],[1,1],[1,1]],
#                      [[1,1],[1,1],[1,1]],
#                      [[1,1],[1,1],[1,1]]],dtype= tf.float32)
# score_v = tf.constant([6,6],dtype=tf.float32)
# b = tf.reshape(query, [-1, 1, 2])
# c = tf.tanh(b + query)
# e = score_v * c
# d = tf.reduce_sum(score_v * c, [2])
# with tf.Session() as sess:
#     # print(sess.run(query))
#     # print(sess.run(b))
#     print(sess.run(h))
#     # print(sess.run(c))
#     # print(sess.run(e))
#     # print(sess.run(d))

a = tf.constant([[[1],[2]],[[3],[4]],[[5],[6]]],dtype=tf.float32)
b = tf.constant([[[1,1,1,1],[2,2,2,2]],[[3,3,3,3],[4,4,4,4]],[[5,5,5,5],[6,6,6,6]]],dtype=tf.float32)
c = a * b
d = tf.reduce_sum(a * b, [1])
e = d.set_shape([None, 4])
with tf.Session() as sess:
    print(sess.run(d))
    print(sess.run(e))
