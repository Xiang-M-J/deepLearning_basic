import datetime
import matplotlib.pyplot as plt
import numpy as np


class Metric:
    def __init__(self, mode="train"):
        if mode == "train":
            self.mode = "train"
            self.train_acc = []
            self.train_loss = []
            self.val_acc = []
            self.val_loss = []
        elif mode == "test":
            self.mode = "test"
            self.test_acc = 0
            self.test_loss = 0
        else:
            print("wrong mode !!! use mode train")
            self.mode = "train"
            self.train_acc = []
            self.train_loss = []
            self.val_acc = []
            self.val_loss = []

    def item(self) -> dict:
        if self.mode == "train":
            data = {"train_acc": self.train_acc, "train_loss": self.train_loss,
                    "val_acc": self.val_acc, "val_loss": self.val_loss}
        else:
            data = {"test_acc": self.test_acc, "test_loss": self.test_loss}
        return data


def plot(metric: dict, model_name: str, result_path: str = "results/", ):
    dpi = 300
    train_acc = metric['train_acc']
    train_loss = metric['train_loss']
    val_loss = metric["val_loss"]
    val_acc = metric['val_acc']
    epoch = np.arange(len(train_acc)) + 1

    plt.figure()
    plt.plot(epoch, train_acc)
    plt.plot(epoch, val_acc)
    plt.legend(["train accuracy", "validation accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.title("train accuracy and validation accuracy")
    plt.savefig(result_path + "images/" + model_name + "_accuracy.png", dpi=dpi)
    plt.show()

    plt.figure()
    plt.plot(epoch, train_loss)
    plt.plot(epoch, val_loss)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss and validation loss")
    plt.savefig(result_path + "images/" + model_name + "_loss.png", dpi=dpi)
    plt.show()


def log(train_metric: Metric, test_metric: Metric, model_name: str, addition: str):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", 'a') as f:
        f.write("========================\t" + date + "\t========================\n")
        f.write(f"model name: \t{model_name}\n")
        f.write(f"addition: \t{addition}\n")
        f.write("train(final): \t\t" + "train loss: {:.4f}\t train accuracy: {:.3f}\t validation loss: {:.4f}\t "
                                       "validation accuracy: {:.3f} \n".format(train_metric.train_loss[-1],
                                                                               train_metric.train_acc[-1],
                                                                               train_metric.val_loss[-1],
                                                                               train_metric.val_acc[-1]))
        f.write("train(max_min): \t" + "train loss: {:.4f}\t train accuracy: {:.3f}\t validation loss: {:.4f}\t "
                                       "validation accuracy: {:.3f} \n".format(min(train_metric.train_loss),
                                                                               max(train_metric.train_acc),
                                                                               min(train_metric.val_loss),
                                                                               max(train_metric.val_acc)))

        f.write("test: \t\t\t\t" + "test loss: \t{:.4f} \t test accuracy:\t {:.3f} \n".format(test_metric.test_loss,
                                                                                              test_metric.test_acc))
        f.write("\n")


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class arguments:
    def __init__(self, lr=1e-3, beta1=0.93, beta2=0.98, batch_size=16, epochs=60,
                 random_seed=46, filter_size=39, dilation_size=8, kernel_size=2, num_class=7, dropout_rate=0.1) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_seed = random_seed
        self.filter_size = filter_size
        self.dilation_size = dilation_size
        self.kernel_size = kernel_size
        self.num_class = num_class
        self.dropout_rate = dropout_rate
