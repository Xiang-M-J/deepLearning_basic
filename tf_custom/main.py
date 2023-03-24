import numpy as np
import os
import tensorflow as tf
from model import TIM_Model
import tensorflow.keras as keras
from utils import arguments, smooth_labels, plot, Metric

# 超参数设置
lr = 1e-3
beta1 = 0.93
beta2 = 0.98
batch_size = 16
epochs = 60
random_seed = 46
filter_size = 39
dilation_size = 8
kernel_size = 2
num_class = 7
dropout_rate = 0.1
spilt_rate = [0.8, 0.1, 0.1]  # 训练集、验证集、测试集分割比例

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')


def train(train_dataset, val_dataset, args):
    model = TIM_Model(args)
    model.build((None, 173, 39))

    lossF = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(lr)
    train_loss = keras.metrics.Mean(name='train_loss')
    train_acc = keras.metrics.CategoricalAccuracy(name="train_acc")
    val_loss = keras.metrics.Mean(name='test_loss')
    val_acc = keras.metrics.CategoricalAccuracy(name='test_acc')
    metric = Metric(mode="train")
    tf.config.experimental_run_functions_eagerly(True)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = lossF(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        predict = tf.argmax(logits, axis=-1)
        label = tf.argmax(y, axis=-1)
        # acc = tf.reduce_sum(predict == label)
        train_acc.update_state(y, logits)
        return loss_value

    @tf.function
    def val_step(x, y):
        val_logits = model(x, training=False)
        val_acc.update_state(y, val_logits)

    for epoch in range(args.epochs):
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()
        # predict_y = model.md(x)
        # Iterate over the batches of the dataset.
        loss = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss = train_step(x_batch_train, y_batch_train)

        for step, (x_val_train, y_val_train) in enumerate(val_dataset):
            val_step(x_val_train, y_val_train)
        metric.train_acc.append(train_acc.result())
        metric.train_loss.append(train_loss.result())

        print("Epoch :{}\t train Loss:{:.4f} \t train Accuracy: {:.3f}\t val Loss:{:.4f} \t val Accuracy:{:.3f} "
              .format(epoch + 1, loss, train_acc.result(), val_loss.result(), val_acc.result()))



if __name__ == "__main__":
    x = np.load("../preprocess/x.npy")
    y = np.load("../preprocess/y.npy")
    Num = x.shape[0]
    random_index = np.random.permutation(range(Num))
    x = x[random_index]
    y = y[random_index]
    train_num = int(Num * spilt_rate[0])
    val_num = int(Num * spilt_rate[1])
    test_num = Num - train_num - val_num

    # ds = tf.data.Dataset.from_tensor_slices((x,y))
    x_train, x_val, x_test = tf.split(
        x,
        num_or_size_splits=[train_num, val_num, test_num],
        axis=0
    )
    y_train, y_val, y_test = tf.split(
        y,
        num_or_size_splits=[train_num, val_num, test_num],
        axis=0
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    args = arguments(lr, beta1, beta2, batch_size, epochs, random_seed, filter_size, dilation_size, kernel_size,
                     num_class, dropout_rate)
    #
    train(train_dataset, val_dataset, args)
    print()
