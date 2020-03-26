import numpy as np
import pandas as pd
import tqdm
from data_utils import load_dataset
import matplotlib.pyplot as plt
from sklearn import neighbors
import time
import random
import a3_mod

__author__ = 'En Xu Li (Thomas)'
__date__ = 'March 24, 2020'




def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def _cast_TF(x):
    """
    change bool type array to one hot encoding with 1 and 0
    Inputs:
        x: (bool type np.array)
    Outputs:
        numpy array with one hot encoding
    """
    return np.where(x==True,1,0)

def _RMSE(x, y):
    return np.sqrt(np.average((x-y)**2))

def _log_likelihood(estimates, actual):
    total = 0
    for i in range(len(estimates)):
        total += actual[i]*np.log(_sigmoid(estimates[i])) + (1-actual[i])*np.log(1 - _sigmoid(estimates[i]))
    return total/len(estimates)

def _Q1_compute_acc(y_test, y_estimates):
    return (y_estimates == y_test).sum() / len(y_test)

def log_reg_GD(dataset='iris', lr_rates=[0.1], method='SGD', total_iter=2000):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

    y_train, y_valid, y_test = _cast_TF(y_train), _cast_TF(y_valid), _cast_TF(y_test)

    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    X = np.ones((len(x_train), len(x_train[0]) + 1))
    X[:, 1:] = x_train

    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test


    test_accuracies = []
    test_logs = []
    neg_log = {}

    for rate in lr_rates:

        w = np.zeros(np.shape(X[0, :]))
        neg_log[rate] = []
        bar = tqdm.tqdm(total=total_iter, desc='Iter', position=0)
        for iteration in range(total_iter):
            bar.update(1)

            estimates = X @ w
            estimates = estimates.reshape(np.shape(y_train))

            if method == 'SGD':
                i = random.randint(0, len(y_train)-1)
                grad_L = (y_train[i] - _sigmoid(estimates[i])) * X[i, :]

            elif method == 'GD':
                grad_L = np.zeros(np.shape(w))
                for i in range(len(y_train)):
                    grad_L += (y_train[i] - _sigmoid(estimates[i])) * X[i, :]

            w = w + (rate*grad_L)
            L = _log_likelihood(estimates, y_train)
            neg_log[rate].append(-L)


        test_estimates = np.dot(X_test, w)
        test_estimates = test_estimates.reshape(np.shape(y_test))
        predictions = np.zeros(np.shape(y_test))
        for i in range(len(predictions)):
            p = _sigmoid(test_estimates[i])
            predictions[i] = (p>=1/2)

        test_accuracies.append(_Q1_compute_acc(y_test, predictions))
        test_logs.append(_log_likelihood(test_estimates, y_test))



    return neg_log, test_accuracies, test_logs

def run_Q1():
    total_iter = 10000
    lr = [0.01,0.001,0.0001,0.00001]
    log_SGD, test_acc_SGD, test_log_SGD = log_reg_GD(dataset='iris', lr_rates=lr, method='SGD', total_iter=total_iter)
    plot(xlabel='Iteration',ylabel='-Log-Likelihood',name='SGD Full Batch',x=list(range(total_iter)),y=[log_SGD[i] for i in log_SGD],legend=['lr = '+str(i) for i in log_SGD])
    print(test_acc_SGD)
    print(test_log_SGD)

    log_GD, test_acc_GD, test_log_GD = log_reg_GD(dataset='iris', lr_rates=lr, method='GD', total_iter=total_iter)
    plot(xlabel='Iteration',ylabel='-Log-Likelihood',name='GD Full Batch',x=list(range(total_iter)),y=[log_GD[i] for i in log_GD],legend=['lr = '+str(i) for i in log_GD])
    print(test_acc_GD)
    print(test_log_GD)

def _update_weights(w, grad_w, lr, dir):
    return w - dir*lr*grad_w

def sgd_mnist(dataset='mnist_small', M=100, batch_size=250, total_iter=5000, lr_rates=[0.0001, 0.001]):

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

    W1 = np.random.randn(M, 784) / np.sqrt(784)
    W2 = np.random.randn(M, M) / np.sqrt(M)
    W3 = np.random.randn(10, M) / np.sqrt(M)
    b1 = np.zeros((M, 1))
    b2 = np.zeros((M, 1))
    b3 = np.zeros((10, 1))


    neg_ll_train, neg_ll_valid, train_acc, val_acc, test_acc, digit_vis = {}, {}, {}, {}, {}, {}


    for rate in lr_rates:

        neg_ll_train[rate] = []
        neg_ll_valid[rate] = []

        min_ll_valid = np.inf
        min_ll_valid_it = 0

        bar = tqdm.tqdm(total=total_iter, desc='Iter')
        for i in range(total_iter):
            bar.update(1)
            nll_valid_fb = a3_mod.negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
            neg_ll_valid[rate].append(nll_valid_fb/len(x_valid))

            idx = np.random.choice(x_train.shape[0], size=batch_size, replace=False)
            mini_batch_x = x_train[idx, :]
            mini_batch_y = y_train[idx, :]

            (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
                a3_mod.nll_gradients(W1, W2, W3, b1, b2, b3, mini_batch_x, mini_batch_y)


            neg_ll_train[rate].append(nll/batch_size)

            W1 = _update_weights(W1, W1_grad, rate, 1)
            W2 = _update_weights(W2, W2_grad, rate, 1)
            W3 = _update_weights(W3, W3_grad, rate, 1)
            b1 = _update_weights(b1, b1_grad, rate, 1)
            b2 = _update_weights(b2, b2_grad, rate, 1)
            b3 = _update_weights(b3, b3_grad, rate, 1)
        train_acc[rate] = _compute_acc(W1, W2, W3, b1, b2, b3, x_train, y_train)
        val_acc[rate] = _compute_acc(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        test_acc[rate] = _compute_acc(W1, W2, W3, b1, b2, b3, x_test, y_test)

        digit_vis[rate] = []
        for i in x_test:
            Fhat = np.exp(a3_mod.forward_pass(W1, W2, W3, b1, b2, b3, i))
            if np.max(Fhat) < 0.5: digit_vis[rate] += [(i,np.argmax(Fhat))]


    return neg_ll_train, neg_ll_valid, train_acc, val_acc, test_acc, digit_vis

def _compute_acc(W1, W2, W3, b1, b2, b3, x, y):
    Fhat = np.exp(a3_mod.forward_pass(W1, W2, W3, b1, b2, b3, x))
    Fhat = np.argmax(Fhat, axis=1)
    y = np.argmax(y, axis=1)
    return (Fhat == y).sum() / len(y)

def run_Q2():
    total_iter = 2000
    neg_ll_train, neg_ll_valid, train_acc, val_acc, test_acc, digit_vis = sgd_mnist(dataset='mnist_small', M=100, batch_size=250, total_iter=total_iter, lr_rates=[0.0001, 0.001])
    for rate in neg_ll_train:
        plot(xlabel='total_iter',ylabel='loss',name='mnist loss curve lr='+str(rate),x=list(range(total_iter)),y=[neg_ll_train[rate],neg_ll_valid[rate]],legend=['train','val'])
        count = 0
        print('total number of unconfident digits: '+str(len(digit_vis[rate])))
        for x,digit in digit_vis[rate]:
            if count >= 5: break
            count += 1
            fig = plt.figure()
            plt.imshow(x.reshape((28,28)), interpolation='none', aspect='equal', cmap='gray')
            plt.savefig('digit_' + str(count) + '_predicted_' + str(digit) + '.png')

    print('Train Acc:')
    print(train_acc)
    print('Train Loss:')
    print([str(rate)+': '+str(neg_ll_train[rate][-1]) for rate in neg_ll_train])
    print('Validation Acc:')
    print(val_acc)
    print('Validation Loss:')
    print([str(rate)+': '+str(neg_ll_valid[rate][-1]) for rate in neg_ll_valid])
    print(test_acc)

def plot(xlabel='',ylabel='',name='fig',x=None,y=None,legend=None):
    """
    plot and figures

    Inputs:
        xlabel: (str) label on x axis
        ylabel: (str) label on y axis
        name: (str) title of the figure
        x: (np.array) x data
        y: (list of np.array) list of y values to plot against x
        legend: (list of str) label on y values

    Outputs:
        None
    """
    fig = plt.figure()
    for i in range(len(y)):
        if legend: plt.plot(x,y[i],label=legend[i])
        else: plt.plot(x,y[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    fig.savefig(name+'.png')

if __name__ == '__main__':
    run_Q1()
    run_Q2()
