import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

def noam(d_model, step, warmup):
    """
    noam scheduler
    """
    fact = min(step ** (-0.5), step * warmup ** (-1.5))
    return fact * (d_model ** (-0.5))

class myscheduler():
    def __init__(self,optimizer,n_epoch,d_model, initial_lr, warmup):
        self.optimizer = optimizer
        self.n_epoch = n_epoch
        self.d_model = d_model  
        self.initial_lr = initial_lr
        self.warmup = warmup
        self.lr = 0

    def step(self, steps):
        self.lr = self.initial_lr  * noam(d_model=self.d_model, step=steps + 1, warmup=self.warmup)
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = self.lr
    def get_lr(self):
        return self.lr
    

model = torchvision.models.resnet18(pretrained=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
scheduler = myscheduler(optimizer,50,512, 0.01, 400) ##初始化scheduler
lr = []
steps = 0
for epoch in range(50):
    for batch in range(100):
        steps += 1
        optimizer.step()
        lr.append(scheduler.get_lr())
    scheduler.step(steps=steps) # 注意 每个epoch 结束， 更新learning rate

print(lr[-1])
plt.plot(np.arange(len(lr)), lr)
plt.show()
