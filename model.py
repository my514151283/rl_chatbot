import tensorflow as tf
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from projection import get_project_funtion
from decoder import dynamic_decoder
from attention_decoder import prepare_attention
from attention_decoder import attention_decoder_fn_train
from attention_decoder import attention_decoder_fn_inference
from model_config import START_WORD_ID,END_WORD_ID
class model(object):
    def __init__(self,
                 embed,  # 词嵌入 [VOCABULARY_COUNT * 200]
                 vocabulary,  # 词汇表 [1 * VOCABULARY_COUNT]
                 vocabulary_count,  # 词汇数
                 num_layers,  # encoder和decoder的层数
                 num_units,  # encoder和decoder的隐藏状态维度
                 learning_rate,
                 max_gradient_norm,
                 max_len,
                 # output_alignments=False
                 ):  # 解码最大长度

        # placeholder通常用于存储数据，用于feed_dict的配合，接收输入数据（如真实的训练样本）用于训练模型等
        # placeholder在训练过程中会不断被赋予新的值，用于批训练，基本上其值是不会轻易进行加减操作
        self.post_string = tf.placeholder(dtype=tf.string, shape=(None, None), name="post_string")  # padding后的post  batch_size * encoder_len
        self.response_string = tf.placeholder(dtype=tf.string, shape=(None, None), name="response_string")  # padding后的response  batch_size * decoder_len
        self.label_string = tf.placeholder(dtype=tf.string, shape=(None, None), name="label_string") # batch_size * decoder_len
        self.post_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="post_len")  # 每条post的长度(padding前) batch_size
        self.response_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="reponse_len")  # 每条response长度(padding前) batch_size

        # tf.get_variable表示创建或返回指定名称的模型变量——共享变量
        self.embed = tf.get_variable(dtype=tf.float32, initializer=embed, name="embed")  # 词嵌入，作为变量训练，VOCABULARY_COUNT * 200
        self.vocabulary = tf.constant(vocabulary, dtype=tf.string)  # 词汇表，VOCABULARY_COUNT

        self.batch_size = tf.shape(self.post_string)[0]
        self.encoder_len = tf.shape(self.post_string)[1]
        self.decoder_len = tf.shape(self.response_string)[1]

        '''
        mask矩阵是一个由0和1组成的矩阵，该矩阵用以指示哪些是真正的数据，哪些是padding
        其中1代表真实数据，0代表padding数据
        [[1. 1. 1. 0. 0.]
         [1. 1. 1. 1. 0.]
         [1. 1. 1. 1. 1.]]
         
        response_len-1:所有长度减去START_WORD所占的位置
        [batch_size * decoder_len]
        tf.cumsum根据列从右往左累计求和
        例如
        右边第一列为原始的[0 0 0]，右边倒数第二列[0+0 0+1 1+0]，右边倒数第三列[0+0+1 0+1+0 1+0+0]
        response_len = [3, 4, 5]    decoder_len = 5
        onehot = [[0. 0. 1. 0. 0.]
                  [0. 0. 0. 1. 0.]
                  [0. 0. 0. 0. 1.]]
        cumsum = [[1. 1. 1. 0. 0.]
                  [1. 1. 1. 1. 0.]
                  [1. 1. 1. 1. 1.]]
        '''
        # self.post_mask = tf.cumsum(tf.one_hot(self.post_len), self.encoder_len), axis=1, reverse=True)
        self.mask = tf.cumsum(tf.one_hot(self.response_len-1, self.decoder_len), axis=1, reverse=True)

        # 将字符(key)转化成id(value)表示的表，默认值为1
        self.string_to_id = MutableHashTable(key_dtype=tf.string,  # 键的类型
                                             value_dtype=tf.int64,  # 值的类型
                                             default_value=1,  # 当检索不到时的默认值
                                             shared_name="string_to_id",  # 如果非空，表将在多个session中以该名字共享
                                             name="string_to_id",  # 操作名
                                             checkpoint=True)  # 如果为True，表能从checkpoint中保存和恢复

        # 将id转化成字符串表示的表，默认值为"_NDW"
        self.id_to_string = MutableHashTable(key_dtype=tf.int64,
                                             value_dtype=tf.string,
                                             default_value="_NDW",
                                             shared_name="id_to_string",
                                             name="id_to_string",
                                             checkpoint=True)

        # 将post和response转化成id表示
        # table.lookup()根据表替换张量值
        self.post_id = self.string_to_id.lookup(self.post_string)  # batch_size * encoder_len
        self.response_id = self.string_to_id.lookup(self.response_string)  # batch_size * decoder_len
        self.label_id = self.string_to_id.lookup(self.label_string)  # batch_size * decoder_len

        # 将post和response转化成嵌入表示
        '''
        tf.nn.embedding_lookup(params, ids，……)根据索引选取一个张量里面对应的元素
        batch_size * encoder_len * embed_size:
            [[[vector_1],
              [vector_2],
              ...
              [vector_encoder_len]],
             [[vector_1],
              [vector_2],
              ...
              [vector_encoder_len]],
             ...,
             [[vector_1],
              [vector_2],
              ...
              [vector_encoder_len]]]
        '''
        self.post_embed = tf.nn.embedding_lookup(embed, self.post_id)  # batch_size * encoder_len * embed_size
        self.response_embed = tf.nn.embedding_lookup(embed, self.response_id)  # batch_size * decoder_len * embed_size

        '''
        Python中对于无需关注其实际含义的变量可以用_代替，这就和for i in range(5)一样，因为这里我们对i并不关心，所以用_代替仅获取值而已
        [LSTMCell(num_units), LSTMCell(num_units)]
        MultiRNNCell用于构建多层循环神经网络
        '''
        # encoder和decoder的层数和维度
        encoder_cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])  # 2层RNN
        decoder_cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])

        projection_fn, loss_fn, inference_fn = get_project_funtion(vocabulary_count)

        # 定义模型的encoder部分
        # tf.variable_scope表示变量所在的命名空间,指定变量的作用域"encoder/变量"
        with tf.variable_scope("encoder"):
            self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(encoder_cell,  # RNN单元
                                                              self.post_embed,  # padding后的post  batch_size * encoder_len * embed_size
                                                              self.post_len,  # post的有效长度  batch_size
                                                              dtype=tf.float32)

            # [batch_size encoder_len num_units] 每个样本每个时间步都对应一个输出
            # self.encoder_output_shape = tf.shape(self.encoder_output)
            # 返回2个LSTMStateTuple(c=array([[batch_size num_units]]),h=array([[batch_size num_units]]))
            # [num_layers(2层) 2(c和h) batch_size num_units] 整个LSTM输出的最终状态，包含C和H，共2层，每个样本都有一个num_units维的状态C和H
            # self.encoder_state_shape = tf.shape(self.encoder_state)

        # 定义模型的decoder部分
        # 训练时decoder
        with tf.variable_scope("decoder"):
            # keys, values, attention_score_fn, attention_construct_fn = \
            #     prepare_attention(self.encoder_output, num_units, reuse=False)
            # decoder_fn_train = attention_decoder_fn_train(self.encoder_state,
            #                                               keys,
            #                                               values,
            #                                               attention_score_fn,
            #                                               attention_construct_fn,
            #                                               output_alignments=output_alignments,
            #                                               decoder_len=self.decoder_len)
            self.decoder_output, self.decoder_state, self.loop_state = dynamic_decoder(decoder_cell,
                                                                                       encoder_state=self.encoder_state,  # num_layers * 2 * batch_size * num_units
                                                                                       input=self.response_embed,
                                                                                       response_len=self.response_len)

            # self.decoder_output_shape = tf.shape(self.decoder_output)  # [batch_size decoder_len num_units]
            # self.decoder_state_shape = tf.shape(self.decoder_state)  # [num_layers 2 batch_size num_units]

            # self.softmaxed_probability = projection_function(self.decoder_output)  # 词汇表softmaxed后的概率 [batch_size decoder_len vovabulary_count]
            # self.maximum_likelihood_id = tf.argmax(self.softmaxed_probability, axis=2)  # [batch_size decoder_len]
            # self.output_string = self.id_to_string.lookup(self.maximum_likelihood_id)
            self.loss, self.avg_loss = loss_fn(self.decoder_output, self.label_id, self.mask)


        '''
        通过tf.variable_scope函数可以控制tf.get_variable函数的语义
        当reuse = True时，这个上下文管理器内所有的tf.get_variable都会直接获取已经创建的变量。如果变量不存在，则会报错
        相反，如果reuse = None或者reuse = False，tf.get_variable将创建新的变量，若同名的变量已经存在则报错
        '''
        # 测试时decoder
        with tf.variable_scope("decoder", reuse=True):
            # keys, values, attention_score_fn, attention_construct_fn = \
            #     prepare_attention(self.encoder_output, num_units, reuse=False)
            # decoder_fn_inference = attention_decoder_fn_inference(self.encoder_state,
            #                                                       keys,
            #                                                       values,
            #                                                       attention_score_fn,
            #                                                       attention_construct_fn,
            #                                                       self.embed,
            #                                                       START_WORD_ID,
            #                                                       END_WORD_ID,
            #                                                       max_len,
            #                                                       vocabulary_count)
            self.inference_output, self.inference_state, self.inference_loop_state  = dynamic_decoder(decoder_cell,
                                                                                                     encoder_state=self.encoder_state,
                                                                                                     projection_function=projection_fn,
                                                                                                     embed=self.embed,
                                                                                                     max_len=max_len)

            self.inference_maximum_likelihood_id = inference_fn(self.inference_output)  # [batch_size decoder_len]
            self.inference_string = self.id_to_string.lookup(self.inference_maximum_likelihood_id)  # [batch_size decoder_len]

        '''
        Variable用于可训练变量，比如网络权重，偏置
        在声明时必须赋予初值，在训练过程中该值很可能会进行不断的加减操作变化
        '''
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        # 获取程序中的全局变量
        self.params = tf.global_variables()
        # 使用自适应优化器——Adam优化算法，创建一个optimizer
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # 根据 decoder_loss 计算 params 梯度，gradients长度等于len(params)
        gradients = tf.gradients(self.loss, self.params)
        # 梯度裁剪
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        # 返回一个执行梯度更新的ops
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

        self.saver = tf.train.Saver()

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))



















