#Author: Alex Chen <alexchensets@gmail.com>

import math
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def forward(x1, x2, x3, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w2_1, w2_2, b1_1, b1_2, b2_1, y):
    z1_1 = w1_1 * x1 + w1_3 * x2 + w1_5 * x3 + b1_1
    z1_2 = w1_2 * x1 + w1_4 * x2 + w1_6 * x3 + b1_2

    a1_1 = sigmoid(z1_1)
    a1_2 = sigmoid(z1_2)

    z2_1 = w2_1 * a1_1 + w2_2 * a1_2 + b2_1
    y_hat = sigmoid(z2_1)
    loss = (1/2) * (y - y_hat) ** 2

    #return old parameters and some values necessary to back prop
    return w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w2_1, w2_2, b1_1, b1_2, b2_1, y, loss, y_hat, a1_1, a1_2

def backward(alpha, x1, x2, x3, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w2_1, w2_2, b1_1, b1_2, b2_1, y, y_hat, a1_1, a1_2):
    dL_db2_1 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * 1
    dL_dw2_1 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * a1_1
    dL_dw2_2 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * a1_2
    dL_db1_1 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * w2_1 * a1_1 * (1 - a1_1) * 1
    dL_db1_2 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * w2_2 * a1_2 * (1 - a1_2) * 1
    dL_dw1_1 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * w2_1 * a1_1 * (1 - a1_1) * x1
    dL_dw1_3 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * w2_1 * a1_1 * (1 - a1_1) * x2
    dL_dw1_5 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * w2_1 * a1_1 * (1 - a1_1) * x3
    dL_dw1_2 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * w2_2 * a1_2 * (1 - a1_2) * x1
    dL_dw1_4 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * w2_2 * a1_2 * (1 - a1_2) * x2
    dL_dw1_6 = 2 * (1/2) * (y - y_hat) * (-1) * y_hat * (1 - y_hat) * w2_2 * a1_2 * (1 - a1_2) * x3
    
    new_b2_1 = b2_1 - alpha * dL_db2_1
    new_w2_1 = w2_1 - alpha * dL_dw2_1
    new_w2_2 = w2_2 - alpha * dL_dw2_2
    new_b1_1 = b1_1 - alpha * dL_db1_1
    new_b1_2 = b1_2 - alpha * dL_db1_2
    new_w1_1 = w1_1 - alpha * dL_dw1_1
    new_w1_2 = w1_2 - alpha * dL_dw1_2
    new_w1_3 = w1_3 - alpha * dL_dw1_3
    new_w1_4 = w1_4 - alpha * dL_dw1_4
    new_w1_5 = w1_5 - alpha * dL_dw1_5
    new_w1_6 = w1_6 - alpha * dL_dw1_6
    
    #return new parameters w and b
    return new_b2_1, new_w2_1, new_w2_2, new_b1_1, new_b1_2, new_w1_1, new_w1_2, new_w1_3, new_w1_4, new_w1_5, new_w1_6

def train():
    #initialization
    x1 = 0.1
    x2 = 0.2
    x3 = 0.3

    w1_1 = 0.11
    w1_2 = 0.12
    w1_3 = 0.13
    w1_4 = 0.14
    w1_5 = 0.15
    w1_6 = 0.16
    w2_1 = 0.21
    w2_2 = 0.22

    b1_1 = 0.11
    b1_2 = 0.12
    b2_1 = 0.21

    y = 0.666
    
    #hyperparameters
    alpha = 0.01 #change this and observe the learning curve
    epoch = 10000
    
    loss_list = []
    for i in range(epoch):
        if i == 0: #first epoch use initialized random parameter
            w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w2_1, w2_2, b1_1, b1_2, b2_1, y, loss, y_hat, a1_1, a1_2 = forward(x1, x2, x3, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w2_1, w2_2, b1_1, b1_2, b2_1, y)
            new_b2_1, new_w2_1, new_w2_2, new_b1_1, new_b1_2, new_w1_1, new_w1_2, new_w1_3, new_w1_4, new_w1_5, new_w1_6 = backward(alpha, x1, x2, x3, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w2_1, w2_2, b1_1, b1_2, b2_1, y, y_hat, a1_1, a1_2)
        else: #afterwards, use new parameter modified by back propagation
            w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w2_1, w2_2, b1_1, b1_2, b2_1, y, loss, y_hat, a1_1, a1_2 = forward(x1, x2, x3, new_w1_1, new_w1_2, new_w1_3, new_w1_4, new_w1_5, new_w1_6, new_w2_1, new_w2_2, new_b1_1, new_b1_2, new_b2_1, y)
            new_b2_1, new_w2_1, new_w2_2, new_b1_1, new_b1_2, new_w1_1, new_w1_2, new_w1_3, new_w1_4, new_w1_5, new_w1_6 = backward(alpha, x1, x2, x3, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w2_1, w2_2, b1_1, b1_2, b2_1, y, y_hat, a1_1, a1_2)
            
        loss_list.append(loss)
        
    plt.plot(loss_list)
    plt.show()

train()
