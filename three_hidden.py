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
data_train = data[0:int(2*m/3)].T
# data_train = data.T
labels_train = data_train[0]
x_train = data_train[1:n]
# print(x_train.shape)
x_train = min_max_normalize(x_train)
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
layer2 = 128
layer3 = 64
layer4 = 32

def init_params():
    w1 = np.random.rand(layer2,784) - 0.5
    b1 = np.random.rand(layer2,1) - 0.5

    w2 = np.random.rand(layer3,layer2) - 0.5
    b2 = np.random.rand(layer3,1) - 0.5

    w3 = np.random.rand(layer4,layer3)- 0.5
    b3 = np.random.rand(layer4, 1) - 0.5

    w4 = np.random.rand(10,layer4) - 0.5
    b4 = np.random.rand(10,1)- 0.5
    return w1, b1, w2, b2, w3, b3, w4, b4

def RelU(z):
    relU = np.maximum(z, 0)
    return relU

def softmax(z):
    e_z = np.exp(z)
    softmax = e_z/sum(e_z)
    return softmax

def sigmoid(z):
    e_negx = np.exp(-z)
    sigmoid = 1 / (1 + e_negx)
    return sigmoid

def deriv_sigmoid(z):
    sig = sigmoid(z)
    return sig*(1-sig)

def forward_prop(x,w1,b1,w2,b2,w3,b3,w4,b4):
    z1 = w1.dot(x) + b1
    a1 = RelU(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)
    z3 = w3.dot(a2) + b3
    a3 = sigmoid(z3)
    z4 = w4.dot(a3) + b4
    a4 = softmax(z4)
    # print(a2.shape)
    
    return z1, a1, z2, a2, z3, a3, z4, a4


def one_hot(y):
    one_hot = np.zeros((y.size, y.max() + 1))
    one_hot[np.arange(y.size), y] = 1
    one_hot = one_hot.T
    return one_hot

def deriv_RelU(z):
    return z > 0
    # return np.where(z > 0, 1, 0)



def back_prop(a1, w1, z1, a2, w2, z2, a3, w3, z3, a4, w4, z4, x, y):
    one_hot_y = one_hot(y)
    
    m = y.size
#     C0 = np.zeroes(10,1)
    dz4 = a4 - one_hot_y
    # print(error2[0])
    dw4 = dz4.dot(a3.T)/m
    db4 = np.sum(dz4)/m

    error3 = w4.T.dot(dz4) * deriv_sigmoid(z3)
    dw3 = error3.dot(a2.T)/m
    db3 = np.sum(error3)/m

    error2 = w3.T.dot(error3) * deriv_sigmoid(z2)
    dw2 = error2.dot(a1.T)/m
    db2 = np.sum(error2)/m

    error1 = w2.T.dot(error2) * deriv_RelU(z1)
    dw1 = error1.dot(x.T)/m
    db1 = np.sum(error1)/m
    
    return dw1, db1, dw2, db2, dw3, db3, dw4, db4


def update_params(w1, b1, w2, b2, w3, b3, w4, b4, dw1, db1, dw2, db2, dw3, db3, dw4, db4, rate):
#     print(f'first : w1, b1, w2, b2 : {w1} {b1} {w2} {b2}')
    w1 = w1 - rate * dw1
    b1 = b1 - rate * db1
    w2 = w2 - rate * dw2
    b2 = b2 - rate * db2
    w3 = w3 - rate * dw3
    b3 = b3 - rate * db3
    w4 = w4 - rate * dw4
    b4 = b4 - rate * db4
    
#     print(f'changed: w1, b1, w2, b2 : {w1} {b1} {w2} {b2}')

    return w1, b1, w2, b2, w3, b3, w4, b4

def get_predictions(a):
    return np.argmax(a, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size, np.sum(y**2-predictions**2) / y.size

def gradient_descent(x, y, iterations, rate, w1=None, b1=None, w2=None, b2=None, w3=None, b3=None, w4=None, b4=None):
    xLoss = []
    yAcc = []
    if not w1:
        w1, b1, w2, b2, w3, b3, w4, b4 = init_params()
    # last_accuracy = 0
    # first = 0
    load = '----------------------------------------------------------|'
    # bar = ''

    # dist = ''
    for iteration in range(iterations):
        z1, a1, z2, a2, z3, a3, z4, a4 = forward_prop(x, w1, b1, w2, b2, w3, b3, w4, b4)
        dw1, db1, dw2, db2, dw3, db3, dw4, db4 = back_prop(a1, w1, z1, a2, w2, z2, a3, w3, z3, a4, w4, z4, x, y)
        w1, b1, w2, b2, w3, b3, w4, b4 = update_params(w1, b1, w2, b2, w3, b3, w4, b4, dw1, db1, dw2, db2, dw3, db3, dw4, db4, rate)

        dist = int((len(load)*iteration)/iterations)
        bar = ['*']*dist
        loader = [*load]
        loader[0:dist] = bar
        load = ''.join(loader)
       
        # p = get_accuracy(get_predictions(a3), y)
        accuracy, loss = get_accuracy(get_predictions(a3), y)
        xLoss.append(loss)
        yAcc.append(accuracy)
        if iteration < iterations-1:
            print(f'{load} accuracy : {accuracy} loss : {loss} {iteration}/{iterations}', end='\r')
        else:
            print(f'{load} accuracy : {accuracy} loss : {loss} {iteration}/{iterations} complete')

    return w1, b1, w2, b2, w3, b3, w4, b4, np.array(xLoss), np.array(yAcc)


def make_predictions(x, w1, b1, w2, b2, w3, b3, w4, b4):
    _, _, _, _, _, a3 = forward_prop(x, w1, b1, w2, b2, w3, b3, w4, b4)
    predictions = get_predictions(a3)
    return predictions

