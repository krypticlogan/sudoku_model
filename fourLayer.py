import pandas as pd
import numpy as np



data = pd.read_csv('train.csv')
# print(data)
data = np.array(data)
m ,n = data.shape
np.random.shuffle(data)


# test.shape = (784,1)
# test = np.array(list(map(lambda el:[el], test)))
# print(test)
# test = transform.resize(test, (1, 784))

def min_max_normalize(set : np.array):
    return (set - np.min(set))/(np.max(set) - np.min(set))

# print(data)
# data_train = data[0:int(2*m/3)].T
data_train = data.T
labels_train = data_train[0]
x_train = data_train[1:n]
# print(x_train.shape)
x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))

data_test = data[int(2*m/3):m].T
labels_test = data_test[0]
x_test = data_test[1:n]
x_test = min_max_normalize(x_test)

# x_test = x_test.join(test.T)



# x_train = x_train / 255.0
# np.random.shuffle(x_train)
# print(f'labels {labels_train}')
# print(x_train)

# x_train.shape

def init_params():
    w1 = np.random.rand(18,784) - 0.5
    b1 = np.random.rand(18,1) - 0.5
    w2 = np.random.rand(12,18) - 0.5
    b2 = np.random.rand(12,1) - 0.5
    w3 = np.random.rand(10,12)- 0.5
    b3 = np.random.rand(10,1)- 0.5
    return w1, b1, w2, b2, w3, b3

def RelU(z):
    relU = np.maximum(z, 0)
    return relU

def softmax(z):
    e_z = np.exp(z)
    softmax = e_z/sum(e_z)
    return softmax

def forward_prop(x,w1,b1,w2,b2,w3,b3):
    z1 = w1.dot(x) + b1
    a1 = RelU(z1)
    z2 = w2.dot(a1) + b2
    a2 = RelU(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    # print(a2.shape)
    
    return z1, a1, z2, a2, z3, a3


def one_hot(y):
    one_hot = np.zeros((y.size, y.max() + 1))
    one_hot[np.arange(y.size), y] = 1
    one_hot = one_hot.T
    return one_hot

def deriv_RelU(z):
    return z > 0
    # return np.where(z > 0, 1, 0)



def back_prop(a1, w1, z1, a2, w2, z2, a3, w3, z3, x, y):
    one_hot_y = one_hot(y)
    
    m = y.size
#     C0 = np.zeroes(10,1)
    dz3 = a3 - one_hot_y
    # print(error2[0])
    dw3 = dz3.dot(a2.T)/m
    db3 = np.sum(dz3)/m

    error2 = w3.T.dot(dz3) * deriv_RelU(z2)
    dw2 = error2.dot(a1.T)/m
    db2 = np.sum(error2)/m

    error1 = w2.T.dot(error2) * deriv_RelU(z1)
    dw1 = error1.dot(x.T)/m
    db1 = np.sum(error1)/m
    
    return dw1, db1, dw2, db2, dw3, db3


def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, rate):
#     print(f'first : w1, b1, w2, b2 : {w1} {b1} {w2} {b2}')
    w1 = w1 - rate * dw1
    b1 = b1 - rate * db1
    w2 = w2 - rate * dw2
    b2 = b2 - rate * db2
    w3 = w3 - rate * dw3
    b3 = b3 - rate * db3
    
#     print(f'changed: w1, b1, w2, b2 : {w1} {b1} {w2} {b2}')

    return w1, b1, w2, b2, w3, b3

def get_predictions(a):
    return np.argmax(a, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iterations, rate):
    w1, b1, w2, b2, w3, b3 = init_params()
    # last_accuracy = 0
    # first = 0
    load = '----------------------------------------------------------|'
    # bar = ''

    # dist = ''
    for iteration in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_prop(x, w1, b1, w2, b2, w3, b3)
        dw1, db1, dw2, db2, dw3, db3 = back_prop(a1, w1, z1, a2, w2, z2, a3, w3, z3, x, y)
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, rate)

        dist = int((len(load)*iteration)/iterations)
        bar = ['*']*dist
        loader = [*load]
        loader[0:dist] = bar
        load = ''.join(loader)
       
        p = get_accuracy(get_predictions(a3), y)
        if iteration < iterations-1:
            print(f'{load} {p}', end='\r')
        else:
            print(f'{load} {p}')

    return w1, b1, w2, b2, w3, b3



def make_predictions(x, w1, b1, w2, b2, w3, b3):
    _, _, _, _, _, a3 = forward_prop(x, w1, b1, w2, b2, w3, b3)
    predictions = get_predictions(a3)
    return predictions

