import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
# print(data)
data = np.array(data)
m ,n = data.shape
np.random.shuffle(data)
print(data.shape)

# test.shape = (784,1)
# test = np.array(list(map(lambda el:[el], test)))
# print(test)
# test = transform.resize(test, (1, 784))

def min_max_normalize(set : np.array):
    return (set - np.min(set))/(np.max(set) - np.min(set))

# data_train = data.T
# print(data)
data_train = data[0:int(2*m/3)].T
labels_train = data_train[0]
x_train = data_train[1:n]
# print(x_train.shape)
x_train = min_max_normalize(x_train).round()

data_test = data[int(2*m/3):m].T
labels_test = data_test[0]
x_test = data_test[1:n]
x_test = min_max_normalize(x_test).round()

# x_test = x_test.join(test.T)



# x_train = x_train / 255.0
# np.random.shuffle(x_train)
# print(f'labels {labels_train}')
# print(x_train)

# x_train.shape
layer2 = 128
layer3 = 64
def init_params():
    print('initializing...')
    w1 = np.random.rand(layer2,784) - 0.5
    b1 = np.random.rand(layer2,1) - 0.5
    w2 = np.random.rand(layer3,layer2) - 0.5
    b2 = np.random.rand(layer3,1) - 0.5
    w3 = np.random.rand(10,layer3)- 0.5
    b3 = np.random.rand(10,1)- 0.5
    print(w1.shape, b1.shape)
    return w1, b1, w2, b2, w3, b3

def RelU(z):
    relU = np.maximum(z, 0)
    return relU

def sigmoid(z):
    e_negx = np.exp(-z)
    sigmoid = 1 / (1 + e_negx)
    return sigmoid

def softmax(z):
    e_z = np.exp(z)
    softmax = e_z/sum(e_z)
    return softmax

def forward_prop(x,w1,b1,w2,b2,w3,b3):
    z1 = w1.dot(x) + b1
    a1 = RelU(z1)
    z2 = w2.dot(a1) + b2
    # a2 = RelU(z2)
    a2 = sigmoid(z2)
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

def deriv_sigmoid(z):
    sig = sigmoid(z)
    deriv = sig*(1-sig)
    return deriv



def back_prop(a1, w1, z1, a2, w2, z2, a3, w3, z3, x, y):
    one_hot_y = one_hot(y)
    
    m = y.size
#     C0 = np.zeroes(10,1)
    dz3 = a3 - one_hot_y
    # print(error2[0])
    dw3 = dz3.dot(a2.T)/m
    db3 = np.sum(dz3)/m

    # error2 = w3.T.dot(dz3) * deriv_RelU(z2)
    error2 = w3.T.dot(dz3) * deriv_sigmoid(z2)
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

    return w1, b1, w2, b2, w3, b3, dw1, dw2, dw3

def get_predictions(a):
    return np.argmax(a, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y)  / y.size, np.sum(y**2-predictions**2) / y.size


# import itertools
def calc_momentum(d_v, d, beta, iteration=None):
    d_v = beta*d_v + (1-beta)*d
    m_hat = d_v/(1- beta**iteration)
    return m_hat

def calc_velocity(d_v, d, beta, iteration=None):
    d_v = beta*d_v + (1-beta)*(np.square(d))
    v_hat = d_v/(1- beta**iteration)
    return v_hat

def adam(dw, mom_b, vel_b, velocity, momentum, iteration):
    epsilon = 1e-8
    # momentum = mom_b*momentum + (1-mom_b)*dw
    m_hat = calc_momentum(momentum, dw, mom_b, iteration)
    # velocity = vel_b*velocity + (1-vel_b)*(dw**2)
    # m_hat = momentum/(1- mom_b**iteration)
    # v_hat = velocity/(1- vel_b**iteration)
    v_hat = calc_velocity(velocity, dw, vel_b, iteration)

    return m_hat/(np.sqrt(v_hat) + epsilon), v_hat, m_hat
    # return m_hat, v_hat

def adam_gradient_descent(x, y, rate, iteration, vw1, mw1, vw2, mw2, vw3, mw3, w1=None, b1=None, w2=None, b2=None, w3=None, b3=None):
    

    # pdw1, pdw2, pdw3 = dw1, dw2, dw3
    z1, a1, z2, a2, z3, a3 = forward_prop(x, w1, b1, w2, b2, w3, b3)
    dw1, db1, dw2, db2, dw3, db3 = back_prop(a1, w1, z1, a2, w2, z2, a3, w3, z3, x, y)
    
    mom_beta = 0.9
    vel_beta = 0.999
 

    dw1, vw1, mw1 = adam(w1, mom_beta, vel_beta, vw1, mw1, iteration)
    dw2, vw2, mw2 = adam(w2, mom_beta, vel_beta, vw2, mw2, iteration)
    dw3, vw3, mw3 = adam(w3, mom_beta, vel_beta, vw3, mw3, iteration)

    w1, b1, w2, b2, w3, b3, _, _, _ = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, rate)

    return w1, b1, w2, b2, w3, b3, a3, vw1, mw1, vw2, mw2, vw3, mw3


def gradient_descent(x, y, rate, iteration, momentum=True, decay=.95, w1=None, b1=None, w2=None, b2=None, w3=None, b3=None, v_dw1=0, v_dw2=0, v_dw3=0, v_db1=0, v_db2=0, v_db3=0):
    

    # pdw1, pdw2, pdw3 = dw1, dw2, dw3
    z1, a1, z2, a2, z3, a3 = forward_prop(x, w1, b1, w2, b2, w3, b3)
    dw1, db1, dw2, db2, dw3, db3 = back_prop(a1, w1, z1, a2, w2, z2, a3, w3, z3, x, y)
    w1, b1, w2, b2, w3, b3, dw1, dw2, dw3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, rate)
    beta = 0.9
    
    if momentum:
        v_dw1 = calc_momentum(v_dw1, dw1, beta, iteration)
        v_dw2 = calc_momentum(v_dw2, dw2, beta, iteration)
        v_dw3 = calc_momentum(v_dw3, dw3, beta, iteration)
        v_db1 = calc_momentum(v_db1, b1, beta, iteration)
        v_db2 = calc_momentum(v_db2, b2, beta, iteration)
        v_db3 = calc_momentum(v_db3, b3, beta, iteration)
    
        # if v_dw == None:
        #     pdw1 = 0
        #     pdw2 = 0
        #     pdw3 = 0     
        # # print(pdw1)
        w1 = w1 - decay*v_dw1
        w2 = w2 - decay*v_dw2
        w3 = w3 - decay*v_dw3
        b1 = b1 - decay*v_db1
        b2 = b2 - decay*v_db2
        b3 = b3 - decay*v_db3
        

    return w1, b1, w2, b2, w3, b3, a3, v_dw1, v_dw2, v_dw3, v_db1, v_db2, v_db3
# `   34QW2`


def minibatchGD(x, y, epochs, rate, batch_size=128,optimizer='', validation=[], validKey='' ,w1=None, b1=None, w2=None, b2=None, w3=None, b3=None, sudoku=None, key=None):
    xLoss = []
    yAcc = []
    validationLoss = []
    validationAcc = []
    # print(y.shape)
    # sudokuLoss = None
    # if sudoku != None:
    # if:
    sudokuLoss = []
    sudokuAcc = []

    load = '----------------------------------------------------------|'
    if not isinstance(w1, np.ndarray):
        w1, b1, w2, b2, w3, b3 = init_params()
    
    
    for epoch in range(epochs):
        pos = 0
        iteration = 0
        vw1 = np.zeros_like(w1) 
        mw1 = np.zeros_like(w1)
        vw2 = np.zeros_like(w2) 
        mw2 = np.zeros_like(w2)
        vw3 = np.zeros_like(w3) 
        mw3 = np.zeros_like(w3)
        vb1 = np.zeros_like(b1) 
        vb2 = np.zeros_like(b2)     
        vb3 = np.zeros_like(b3) 
        # 
        while pos < batch_size:
            iteration+=1
            batchX = x.T[pos:pos+batch_size].T
            batchY = y.T[pos:pos+batch_size].T
            # batchYpreds = 
            # display(x.shape, batchX.shape)
            # display(y.shape, batchY.shape)
            pos = pos + batch_size
            dist = int((len(load)*epoch)/epochs)
            bar = ['*']*dist
            loader = [*load]
            loader[0:dist] = bar
            load = ''.join(loader)

            # batch = x.T[pos:epoch*batch_size].T
            if optimizer == "adam":
                w1, b1, w2, b2, w3, b3, a3, vw1, mw1, vw2, mw2, vw3, mw3 = adam_gradient_descent(x=batchX, y=batchY, rate=rate, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3,vw1=vw1, mw1=mw1, vw2=vw2, mw2=mw2, vw3=vw3, mw3=mw3, iteration=iteration)
            else:
                w1, b1, w2, b2, w3, b3, a3, vw1, vw2, vw3, vb1, vb2, vb3 = gradient_descent(x=batchX, y=batchY, rate=rate, iteration=iteration, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3, v_dw1=vw1, v_dw2=vw2, v_dw3=vw3, v_db1=vb1, v_db2=vb2, v_db3=vb3)
            # print(a3.shape, batchY.shape)
            accuracy, loss = get_accuracy(get_predictions(a3), batchY)
            xLoss.append(loss)
            yAcc.append(accuracy)
            
            # print('val', validation.shape, 'w1', w1.shape, 'b1', b1.shape)
            
            acc, val_loss = get_accuracy(make_predictions(validation, w1, b1, w2, b2, w3, b3), validKey)
            validationLoss.append(val_loss)
            validationAcc.append(acc)

            if sudoku is not None:
                sudoku_acc, sudoku_loss = get_accuracy(make_predictions(sudoku, w1, b1, w2, b2, w3, b3), key)
                sudokuLoss.append(sudoku_loss)
                sudokuAcc.append(sudoku_acc)



            if epoch < epochs-1:
                print('\r', f'[{epoch}/{epochs}] {load} {accuracy} {loss}', end='')
            else:
                print(f'{load} {accuracy} {loss} complete')

    return w1, b1, w2, b2, w3, b3, np.array(xLoss), np.array(yAcc), np.array(validationLoss), np.array(validationAcc), np.array(sudokuLoss), np.array(sudokuAcc)

def make_predictions(x, w1, b1, w2, b2, w3, b3):
    _, _, _, _, _, a3 = forward_prop(x, w1, b1, w2, b2, w3, b3)
    predictions = get_predictions(a3)
    return predictions

# minibatchGD(x_train, y_label,2,2,2)

def get_weights():
    w1, b1, w2, b2, w3, b3, = np.genfromtxt('data/params/w1.csv', delimiter=',')[1:], np.genfromtxt('data/params/b1.csv', delimiter=',')[1:], np.genfromtxt('data/params/w2.csv', delimiter=',')[1:], np.genfromtxt('data/params/b2.csv', delimiter=',')[1:], np.genfromtxt('data/params/w3.csv', delimiter=',')[1:], np.genfromtxt('data/params/b3.csv', delimiter=',')[1:]

    b1 = b1.reshape(-1,1)
    b2 = b2.reshape(-1,1)
    b3 = b3.reshape(-1,1)
    return w1, b1, w2, b2, w3, b3

