import tensorflow as tf
from model_config import START_WORD_ID, END_WORD_ID

"""
动态RNN复写函数模板 
def loop_fn(time,  # 初始为0，第time个时间步之前的处理，标量
            cell_output,  # 上一个时间步的输出
            cell_state,  # RNNCells长时记忆
            loop_state):  # 存放一些循环信息，例如是否已完成(保存了上个时间步执行后是否已经结束)或者注意力系数(如果输出 alignments，还保存了存有 alignments 的 TensorArray)
    return (next_done,  # 是否已完成，boolean
            next_cell_input,  # 下个时间步的输入，一般是输入加上注意力的拼接
            next_cell_state,  # 下个时间步的状态，RNN长时记忆，在第一个时间步时初始化，之后都不用去管它
            emit_output,  # 模型的实际输出
            next_loop_state)  # 存放一些循环的信息
"""
def dynamic_decoder(cell,  # RNNCell
                    encoder_state,  # encoder最后输出状态 [[array(batch_size num_units),array(batch_size num_units)],[array(batch_size num_units),array(batch_size num_units)]]
                    input=None,  # 训练时，以response_embed为输入 [batch_size decoder_len embed_size]
                    response_len=None,  # 回复的长度列表 [batch_size]
                    projection_function=None,  # 将decoder输出映射到词汇表的函数
                    embed=None,  # [VOCABULARY_COUNT * embed_size(200)]
                    max_len=None):  # 解码最大长度

    with tf.name_scope("dynamic_decoder"):
        if input is not None:  # 训练时
            dtype = input.dtype  # tf.float32
            decoder_len = tf.shape(input)[1]  # decoder_len, 解码最大时间步
            '''
            交换张量的不同维度
            # 4*2*3
            input = [[[0,1,2],[9,8,7]]
                     [[3,4,5],[0,0,0]]
                     [[6,7,8],[6,5,4]]
                     [[9,0,1],[3,2,1]]]
            # 2*4*3
            tf.transpose(input, perm=[1, 0, 2])=
                    [[[0,1,2]
                      [3,4,5]
                      [6,7,8]
                      [9,0,1]]
                     [[9,8,7]
                      [0,0,0]
                      [6,5,4]
                      [3,2,1]]]
            '''
            input = tf.transpose(input, perm=[1, 0, 2])  # [decoder_len batch_size embed_size]
            # 大小为decoder_len的TensorArray
            input_tensorarray = tf.TensorArray(dtype=dtype, size=decoder_len, clear_after_read=False)
            # 输入Tensor，输出一个新的TensorArray对象[TensorArray_1(batch_size embedding_size),TensorArray_2,...TensorArray_decoder_len]
            input_tensorarray = input_tensorarray.unstack(input)  # decoder_len*[batch_size embedding_size]

######### 动态RNN复写函数 ##############################################################################################
        def loop_fn(time,
                    cell_output,
                    cell_state,
                    loop_state):
            '''
            loop_fn是一个在RNN的相邻时间步之间被调用的函数
            函数的总体调用过程为：
                1. 初始时刻，先调用一次loop_fn，获取第一个时间步的cell的输入，loop_fn中进行读取初始时刻的输入。
                2. 进行cell自环　(output, cell_state) = cell(next_input, state)
                3. 在 t 时刻 RNN 计算结束时，cell 有一组输出 cell_output 和状态 cell_state，都是 tensor；
                4. 到 t+1 时刻开始进行计算之前，loop_fn 被调用，调用的形式为
                   loop_fn( t, cell_output, cell_state, loop_state)
                   而被期待的输出为：(finished, next_input, initial_state, emit_output, loop_state)
                5. RNN 采用 loop_fn 返回的 next_input 作为输入，initial_state 作为状态，计算得到新的输出
                  在每次执行（output， cell_state） =  cell(next_input, state)后，执行 loop_fn() 进行数据的准备和处理。
            用更加通俗易懂的语言描述如下：
                1.利用 loop_fn 计算 time=0 时的一系列初始变量
                2.进入循环，在batch内全部样本 “finish” 时才结束
                3.循环内，由 cell 计算新的 output 和 state
                4.循环内，由 loop_fn 处理 output 和 state，决定本轮最终的输出 emit 和状态 next_state，以及下一轮的输入 next_input
            也就是说，实际上即使只有 cell 也能完成RNN的功能，但是 loop_fn 的存在允许做进一步处理。
            值得注意的是，如果 cell 是多个的（tuple型），那么这里最终的 emit_ta 和 state 也将是复数的。
            :param time: 初始为0，标量
            :param cell_output: 上一个时间步的输出  batch_size * decoder_len
            :param cell_state: RNNCells长时记忆
            :param loop_state: 存放一些循环信息，
                               例如上个时间步执行后是否已经结束
                               注意力系数(如果输出 alignments，还保存了存有 alignments 的 TensorArray)
            :return:next_done,  # 是否已完成，boolean  [batch_size]
                    next_cell_input,  # 下个时间步的输入，一般是输入加上注意力的拼接  [batch_size embedding_size]
                    next_cell_state,  # 下个时间步的状态，在第一个时间步时初始化，之后都不用去管它  [num_layers 2(c,h) batch_size num_units]
                    emit_output,  # 模型的实际输出  batch_size * decoder_len
                    next_loop_state)  # 存放一些循环的信息
            '''


            # 在训练的模式下 ###########################################################################################
            if input is not None:
                if cell_state is None:  # 第0个时间步之前的处理
                    emit_output = None  # 第0个时间步之前是没有输出的
                    cell_state = encoder_state  # coder赋值encoder的最后状态 [num_layers 2(c,h) batch_size num_units]
                    next_cell_state = cell_state  # 将cell状态一直传递下去
                    next_cell_input = input_tensorarray.read(0)  # 读取第0个时间步的输入，即第一列，GO_ID [batch_size embedding_size]
                    next_loop_state = loop_state  # 将循环状态信息一直传递下去，如果有必要可以从里面存取一些信息
                    # 与resonse_len长度相同，[True False False True ...]
                    next_done = time >= response_len  # 如果是第response_len个时间步之前的处理，说明已经解码完成了
                    # 这里可以再加入一些初始化信息
                else:  # 之后的时间步的处理
                    emit_output = cell_output  # 这里的输出并没有做任何加工
                    next_cell_state = cell_state  # 将cell状态一直传递下去
                    '''
                    tf.cond(pred, true_fn = None,false_fn = None)
                    pred为True，返回true_fn，否则返回 false_fn
                    '''
                    next_cell_input = tf.cond(
                        # tf.equal(a,b)比较张量对应的元素，如果相等就返回True，否则返回False，返回的张量维度和a是一样的
                        tf.equal(time, decoder_len),
                        lambda: tf.zeros_like(input_tensorarray.read(0), dtype=dtype),  # 返回一个和input_tensorarray.read(0)相同，全0填充的张量，batch_size * embedding_size
                        lambda: input_tensorarray.read(time),
                    )
                    next_done = time >= response_len  # 如果是第response_len个时间步之前的处理，说明已经解码完成了
                    next_loop_state = loop_state  # 将循环状态信息一直传递下去，如果有必要可以从里面存取一些信息

            # 在推导的模式下 ###########################################################################################
            else:

                if cell_state is None:  # 第0个时间步之前的处理
                    emit_output = None  # 第0个时间步之前是没有输出的
                    cell_state = encoder_state  # decoder赋值encoder的最后状态
                    next_cell_state = cell_state  # 将cell状态一直传递下去
                    batch_size = tf.shape(cell_state)[2]
                    # [2, 2, 2,...]
                    next_cell_input_id = tf.ones([batch_size], dtype=tf.int32) * START_WORD_ID  # 第一步的输入为起始词 batch_size
                    '''
                    用一维索引数组next_cell_input_id，将张量中对应索引的词向量提取出来
                    next_cell_input=[[0 0 0 0 ...0]
                                     ...
                                     [0 0 0 0 ...0]]
                    '''
                    next_cell_input = tf.gather(embed, next_cell_input_id)  # 第一步的输入 [batch_size embedding_size]
                    next_loop_state = loop_state  # 将循环状态信息一直传递下去，如果有必要可以从里面存取一些信息
                    next_done = tf.zeros([batch_size], dtype=tf.bool)  # [False False ... False] 每个样本next_done的取值列表
                    # 这里可以再加入一些初始化信息
                else:  # 之后的时间步的处理
                    emit_output = cell_output  # 这里的输出并没有做任何加工
                    next_cell_state = cell_state  # 将cell状态一直传递下去
                    batch_size = tf.shape(cell_state)[2]
                    softmaxed_probability = projection_function(emit_output)  # 词汇表softmaxed后的概率 [batch_size vovabulary_count]
                    '''
                    每个样本，当前生成的单词应取的对应索引
                    [1, 3, ...] 第一个样本vocabulary[1]概率最大，第二个样本vocabulary[3]最大...
                    查找某个tensor对象在某一维上的其数据最大值所在的索引值(axis=1表示行)
                    '''
                    maximum_likelihood_id = tf.argmax(softmaxed_probability, axis=1)  # [batch_size]
                    '''
                    如果当前最大概率ID是END_WORD_ID,则为True,否则为Flase
                    [False, True, ...]
                    '''
                    done = tf.equal(maximum_likelihood_id, END_WORD_ID)  # [batch_size]
                    next_cell_input = tf.gather(embed, maximum_likelihood_id)  # [batch_size, embedding_size]
                    # 如果time == max_len，则next_done全部设为False(算损失时候超过句子长度就不算了)；如果不等，则将next_done = done
                    next_done = tf.cond(
                        tf.equal(time, max_len),
                        lambda: tf.ones([batch_size], dtype=tf.bool),
                        lambda: done,
                    )
                    next_loop_state = loop_state


            return (next_done, next_cell_input, next_cell_state, emit_output, next_loop_state)
########################################################################################################################

        output_tensorarray, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn, scope="decoder_rnn")


        output = output_tensorarray.stack()  # [decoder_len batch_size num_units]
        output = tf.transpose(output, perm=[1, 0, 2])  # [batch_size decoder_len num_units]

        return output, final_state, final_loop_state