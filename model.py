import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
# print(data)
data = np.array(data)
m ,n = data.shape
np.random.shuffle(data)



def min_max_normalize(set : np.array):
    return (set - np.min(set))/(np.max(set) - np.min(set))

# print(data)z
data_train = data[0:2000].T
labels_train = data_train[0]
x_train = data_train[1:n]
x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))

data_test = data[2000:m].T
labels_test = data_test[0]
x_test = data_test[1:n]
x_test = min_max_normalize(x_test)


# x_train = x_train / 255.0
# np.random.shuffle(x_train)
# print(f'labels {labels_train}')
# print(x_train)

# x_train.shape

def init_params():
    w1 = np.random.rand(12,784) - 0.5
    b1 = np.random.rand(12,1) - 0.5
    w2 = np.random.rand(10,12)- 0.5
#     print(w2.shape)
    b2 = np.random.rand(10,1)- 0.5
    return w1, b1, w2, b2

# w1, b1, w2, b2 = init_params()

def RelU(z):
    relU = np.maximum(z, 0)
#     print(f'relu : {relU}')
    return relU

def softmax(z):
    # z = z - np.max(z)
    e_z = np.exp(z)
    softmax = e_z/sum(e_z)

    return softmax

def forward_prop(x,w1,b1,w2,b2):
    # print(b1.shape)
    z1 = w1.dot(x) + b1
    a1 = RelU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    # print(a2.shape)
    
    return z1, a1, z2, a2


def one_hot(y):
    one_hot = np.zeros((y.size, y.max() + 1))
    one_hot[np.arange(y.size), y] = 1
    one_hot = one_hot.T
    return one_hot

def deriv_RelU(z):
    return z > 0
    # return np.where(z > 0, 1, 0)



def back_prop(z1,a1,w2,a2, x, y):
    one_hot_y = one_hot(y)
    
    m = y.size
#     C0 = np.zeroes(10,1)
    error2 = a2 - one_hot_y
    # print(error2[0])
    dw2 = error2.dot(a1.T)/m
    db2 = np.sum(error2)/m

    error1 = w2.T.dot(error2) * deriv_RelU(z1)
    dw1 = error1.dot(x.T)/m
    db1 = np.sum(error1)/m
    
    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, rate):
#     print(f'first : w1, b1, w2, b2 : {w1} {b1} {w2} {b2}')
    w1 = w1 - rate * dw1
    b1 = b1 - rate * db1
    w2 = w2 - rate * dw2
    b2 = b2 - rate * db2
    
#     print(f'changed: w1, b1, w2, b2 : {w1} {b1} {w2} {b2}')

    return w1, b1, w2, b2

def get_predictions(a):
    return np.argmax(a, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iterations, rate):
    w1, b1, w2, b2 = init_params()
    # last_accuracy = 0
    # first = 0
    load = '----------------------------------------------------------|'
    # bar = ''

    # dist = ''
    for iteration in range(iterations):
        z1, a1, z2, a2 = forward_prop(x, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = back_prop(z1, a1, w2, a2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, rate)
        dist = int((len(load)*iteration)/iterations)
        bar = ['*']*dist
        loader = [*load]
        # print(loader)
        loader[0:dist] = bar
        # loader.append()
        load = ''.join(loader)
        # if iteration == 1:
                # first = p
                
        # if iteration % 50 == 0:
        #     print(f'iteration : {iteration}')
      
        p = get_accuracy(get_predictions(a2), y)
        # print(load)
        if iteration < iterations-1:
            print(f'{load} {p}', end='\r')
        else:
            print(f'{load} {p}')

        
        
            # last_accuracy = p
    # print(f'improvement : {last_accuracy - first}')
    return w1, b1, w2, b2



def make_predictions(x, w1, b1, w2, b2):
    _, _, _, a2 = forward_prop(x, w1, b1, w2, b2)
    predictions = get_predictions(a2)
    return predictions


# test_data = pd.read_csv('test.csv')
# test_data = np.array(test_data)
# m, n = test_data.shape
# test_data = test_data.T
# test_labels = test_data[0]
# test_x = test_data[0:n]/255.0


