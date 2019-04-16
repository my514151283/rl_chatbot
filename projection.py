import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers

def get_project_funtion(vocabulary_count):  # 返回一个能将隐藏状态映射函数

    # 将decoder输出映射到词汇表的函数
    def project_fn(input):
        '''
        如果input是batch_size*num_units,创建权重矩阵 W:num_units*vocabulary_count
        output = input * W  batch_size * vocabulary_count
        如果input是batch_size, decoder-len, num_units，则output是batch_size, decoder-len, vocabulary_count
        '''
        output = layers.linear(input, vocabulary_count, scope="projection_layer")  # batch_size * vocabulary_count
        softmaxed_probability = tf.nn.softmax(output)  # batch_size * vocabulary_count
        return softmaxed_probability

    def loss_fn(decoder_output, label_id, mask):
        '''
        :param decoder_output: [batch_size decoder_len num_units]
        :param label_id: batch_size,decoder_len
        :param mask: [batch_size,decoder_len]
        :return:
        '''
        with tf.variable_scope("decoder_rnn"):
            softmaxed_probability = layers.linear(decoder_output, vocabulary_count, scope="projection_layer")  # batch_size decoder_len vocabulary_count
            logits = tf.reshape(softmaxed_probability, [-1, vocabulary_count])  # 二维[batch_size*decoder_len, vovabulary_count]
            labels = tf.reshape(label_id, [-1])  # [batch_size*decoder_len]
            label_mask = tf.reshape(mask, [-1])  # [batch_size*decoder_len]
            '''
            logits是神经网络输出层的输出，shape为[batch_size,num_classes]
            label是一个一维向量，长度为batch_size，每个元素取值区间是[0,num_classes)，其实每一个值就是代表了batch中对应样本的类别
            tf.nn.sparse_softmax_cross_entropy_with_logits该函数先计算logits的softmax值，再计算softmax与label的交叉熵损失
            因此传入的logits无须提前softmax
            '''
            local_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [batch_size*decoder_len]
            total_size = tf.reduce_sum(label_mask)  # batch_size个response的总长度（不算padding部分）
            total_size += 1e-12  # 避免总长度为0
            loss = tf.reduce_sum(local_loss)  # batch_size个response的总损失
            avg_loss = loss / total_size  # 每个单词的平均损失
            return loss, avg_loss

    def inference_fn(inference_output):
        with tf.variable_scope("decoder_rnn"):
            output = layers.linear(inference_output, vocabulary_count, scope="projection_layer")
            inference_softmaxed_probability = tf.nn.softmax(output)  # 词汇表softmaxed后的概率 [batch_size decoder_len vovabulary_count]
            inference_maximum_likelihood_id = tf.argmax(inference_softmaxed_probability, axis=2)
            return inference_maximum_likelihood_id

    return project_fn, loss_fn, inference_fn
