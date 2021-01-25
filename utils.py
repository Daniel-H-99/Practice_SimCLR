import random
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

# data manager for recording, saving, and plotting
class AverageMeter(object):
    def __init__(self, args, name='noname', save_all=False, surfix='.', x_label=None):
        self.args = args
        self.name = name
        self.save_all = save_all
        self.surfix = surfix
        self.path = os.path.join(args.path, args.result_dir, args.name, surfix)
        self.x_label = x_label
        self.reset()
    def reset(self):
        self.max = - 100000000
        self.min = 100000000
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.save_all:
            self.data = []
    def load_array(self, data):
        self.max = max(data)
        self.min = min(data)
        self.val = data[-1]
        self.sum = sum(data)
        self.count = len(data)
        if self.save_all:
            self.data.extend(data)
    def update(self, val, weight=1):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
        if self.save_all:
            self.data.append(val)
        is_max, is_min = False, False
        if val > self.max:
            self.max = val
            is_max = True
        if val < self.min:
            self.min = val
            is_min = True
        return (is_max, is_min)
    def save(self):
        with open(os.path.join(self.path, "{}.txt".format(self.name)), "w") as file:
            file.write("max: {0:.4f}\nmin: {1:.4f}".format(self.max, self.min))
        if self.save_all:
            np.savetxt(os.path.join(self.path, "{}.csv".format(self.name)), self.data, delimiter=',')
    def plot(self, scatter=False):
        assert self.save_all
        plot_1D(self.args, self.data, scatter=scatter, surfix=self.surfix, name=self.name, x_label=self.x_label, y_label=self.name)
    def plot_over(self, rhs, scatter=True, x_label=True, y_label=True, title=None, save=True):
        assert self.save_all and rhs.save_all
        plot_2D(self.args, self.data, rhs.data, scatter=scatter, surfix=self.surfix, name=self.name, x_label=self.x_label, y_label=self.name)
        
        
# convert idice of words to real words
def seq2sen(batch, vocab):
    sen_list = []

    for seq in batch:
        seq_strip = seq[:seq.index(1)+1]
        sen = ' '.join([vocab.itow(token) for token in seq_strip[1:-1]])
        sen_list.append(sen)

    return sen_list

# shuffle source and target lists in paired manner
def shuffle_list(src, tgt):
    index = list(range(len(src)))
    random.shuffle(index)

    shuffle_src = []
    shuffle_tgt = []

    for i in index:
        shuffle_src.append(src[i])
        shuffle_tgt.append(tgt[i])

    return shuffle_src, shuffle_tgt

# simple metric whether each predicted words match to original ones
def val_check(pred, ans):
    # pred, ans: (batch x length)
    batch, length = pred.shape
    num_correct = (pred == ans).sum()
    total = batch * length
    
    return num_correct, total

# save data, such as model, optimizer
def save(args, surfix, data):
    torch.save(data, os.path.join(args.path, args.ckpt_dir, args.name, "{}.pt".format(surfix)))

# load data, such as model, optimizer
def load(args, surfix, map_location='cpu'):
    return torch.load(os.path.join(args.path, args.ckpt_dir, "{}.pt".format(surfix)), map_location=map_location)

# draw 1D plot
def plot_1D(args, x, scatter=True, surfix='.', name='noname', x_label=None, y_label=None):
    if scatter:
        plot = plt.scatter(range(1, 1+ len(x)), x)
    else:
        plot = plt.plot(range(1, 1 + len(x)), x)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(os.path.join(args.path, args.result_dir, args.name, surfix, "{}.jpg".format(name)))
    plt.close(plt.gcf())
    
# draw 2D plot
def plot_2D(args, x, y, scatter=True, surfix='.', name='noname', x_label=None, y_label=None):
    assert len(x) == len(y)
    if scatter:
        plot = plt.scatter(x, y)
    else:
        plot = plt.plot(x, y)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(os.path.join(args.path, args.result_dir, args.name, surfix, "{}.jpg".format(name)))
    plt.close(plt.gcf())

def top_n_accuracy_score(y_prob, y_true, n=5, as_percentage=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = torch.argsort(y_prob, dim=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if as_percentage:
        return counter * 1.0 / num_obs * 100
    else:
        return counter * 1.0 / num_obs