import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D, BatchNormalization, SpatialDropout1D, ReLU, Add, GlobalAveragePooling1D, \
    Concatenate, Dense, Input
from tensorflow.keras.activations import sigmoid


def Temporal_Aware_Block(x, dilation_rate=8, filters=39, kernel_size=2, rate=0, name='TAB'):
    input_x = x
    x1 = Conv1D(filters=filters, kernel_size=kernel_size,  # filters: 整数，输出空间的维度
                dilation_rate=dilation_rate, padding='causal')(x)  # dilation_rate 用于膨胀卷积的膨胀率
    # padding='causal' 表示因果（膨胀）卷积，加上padding='causal'后，输出的特征维度不会减小
    # 关于因果卷积：https://zhuanlan.zhihu.com/p/231108835
    x1 = BatchNormalization(trainable=True, axis=-1)(x1)
    x1 = ReLU()(x1)
    o1 = SpatialDropout1D(rate)(x1)

    x2 = Conv1D(filters=filters, kernel_size=kernel_size,
                dilation_rate=dilation_rate, padding='causal')(o1)
    x2 = BatchNormalization(trainable=True, axis=-1)(x2)
    x2 = ReLU()(x2)
    o2 = SpatialDropout1D(rate)(x2)

    if input_x.shape[-1] != o2.shape[-1]:
        input_x = Conv1D(filters=filters, kernel_size=1, padding='same')(input_x)

    o2 = sigmoid(o2)
    # output = Lambda(lambda x: tf.multiply(x[0], x[1]))([input_x, o2]) # 原始输入与输出点乘(残差结构)
    output = tf.multiply(input_x, o2)
    return output


# x = tf.random.normal([10,188,39])
# print(Temporal_Aware_Block(x=x))
class TIMNet:
    def __init__(self, dilation=8, filters=39, kernel_size=2, dropout_rate=0.1, return_sequences=True, name="TIM"):
        self.name = name
        self.return_sequences = return_sequences
        self.dilation = dilation
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        if self.dilation is None:
            self.dilation = 8
        input = x
        input_r = tf.reverse(x, axis=[1])  # axis = 1 按行倒转

        conv11 = Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                        dilation_rate=1, padding="causal")(input)
        conv11_r = Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                          dilation_rate=1, padding="causal")(input_r)
        skip_out = conv11
        skip_out_r = conv11_r
        skip_concatenate = []
        for i in range(self.dilation):
            dilation_rate = 2 ** i
            skip_out = Temporal_Aware_Block(skip_out, dilation_rate,
                                            self.filters, self.kernel_size,
                                            self.dropout_rate, )  # 第0个位置和第i个位置与kernel（假设size=2）卷积
            skip_out_r = Temporal_Aware_Block(skip_out_r, dilation_rate,
                                              self.filters, self.kernel_size, self.dropout_rate, )

            skip_add = Add(name="add_" + str(i))([skip_out, skip_out_r])
            # Add将[skip_out, skip_out_r]的输出相加 skip_add相当于论文中的g_i
            skip_add = GlobalAveragePooling1D()(skip_add)  # 时态数据的全局平均池化操作 (None, 188, 39) --> (None, 39)
            # GlobalAveragePooling1D()允许模型以尽可能最简单的方式处理可变长度的输入。
            skip_add = tf.expand_dims(skip_add, axis=1)  # 增加维度(None, 39) --> (None, 1, 39)
            skip_concatenate.append(skip_add)
        output = skip_concatenate[0]
        for i, item in enumerate(skip_concatenate):
            if i == 0:
                continue
            output = Concatenate(axis=-2)([output, item])  # 串联操作 [(None, 39) (None, 39)] = [(None, 2, 39)]
        return output  # 由于dilations = 8，所以x的维度为(None, 8, 39)


# x = tf.random.normal([10,188,39])
# TIM()(x)
# o2 = TIMNET()(x)
# print(o1)
# print(o2)
# print(tf.math.equal(o1,o2))
class FusionWeightLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super(FusionWeightLayer, self).build(input_shape=input_shape)
        self.w = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="uniform",
            trainable=True,
            name="weight"
        )

    def call(self, x):
        x_T = tf.transpose(x, [0, 2, 1])  # (None, 8, 39) -> (None, 39, 8)
        x = K.dot(x_T, self.w)  # x.shape = (None, 39, 1)
        x = tf.squeeze(x, axis=-1)  # 删除最后一个维度  (None, 39, 1) -> (None, 39)
        return x

    def get_config(self):
        config = super(FusionWeightLayer, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class TIM_Model(keras.Model):
    def __init__(self, args, name="TIM_Model"):
        super(TIM_Model, self).__init__()
        # self.inputs = Input(shape=())
        self.TIMNet = TIMNet(dilation=args.dilation_size, filters=args.filter_size, kernel_size=args.kernel_size,
                             dropout_rate=args.dropout_rate, return_sequences=True, name="TIM")
        self.Fusion = FusionWeightLayer()
        self.Classifier = Dense(args.num_class, activation="softmax")
        print("Model build success")

    def call(self, x, training=None, mask=None):
        x = self.TIMNet(x)
        x = self.Fusion(x)
        x = self.Classifier(x)
        return x
