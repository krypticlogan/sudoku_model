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
data_train = data.T
labels_train = data_train[0]
x_train = data_train[1:n]
x_train = min_max_normalize(x_train).round()

data_test = data[2000:m].T
labels_test = data_test[0]
x_test = data_test[1:n]
x_test = min_max_normalize(x_test).round()


# x_train = x_train / 255.0
# np.random.shuffle(x_train)
# print(f'labels {labels_train}')
# print(x_train)

# x_train.shape

def init_params():
    w1 = np.random.rand(32,784) - 0.5
    b1 = np.random.rand(32,1) - 0.5
    w2 = np.random.rand(10,32)- 0.5
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

    return w1, b1, w2, b2, dw1, dw2

def get_predictions(a):
    return np.argmax(a, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size, np.sum(y**2-predictions**2) / y.size

# def gradient_descent(x, y, iterations, rate, w1=None, b1=None, w2=None, b2=None):

#     if not isinstance(w1, np.ndarray):
#         w1, b1, w2, b2 = init_params()
#     # last_accuracy = 0
#     # first = 0
#     load = '----------------------------------------------------------|'
#     # bar = ''

#     # dist = ''
#     for iteration in range(iterations):
#         z1, a1, z2, a2 = forward_prop(x, w1, b1, w2, b2)
#         dw1, db1, dw2, db2 = back_prop(z1, a1, w2, a2, x, y)
#         w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, rate)
#         dist = int((len(load)*iteration)/iterations)
#         bar = ['*']*dist
#         loader = [*load]
#         # print(loader)
#         loader[0:dist] = bar
#         # loader.append()
#         load = ''.join(loader)
#         # if iteration == 1:
#                 # first = p
                
#         # if iteration % 50 == 0:
#         #     print(f'iteration : {iteration}')
#         preds = get_predictions(a2)
#         # loss = np.sum(preds)
#         p, loss = get_accuracy(preds, y)
#         # print(load)
#         if iteration < iterations-1:
#             print(f'{load} {p} {loss}', end='\r')
#         else:
#             print(f'{load} {p} {loss}')

        
        
#             # last_accuracy = p
#     # print(f'improvement : {last_accuracy - first}')
#     return w1, b1, w2, b2
def gradient_descent(x, y, rate, momentum=False, decay=None, w1=None, b1=None, w2=None, b2=None, dw1=None, dw2=None):
    pdw1, pdw2 = dw1, dw2
    z1, a1, z2, a2= forward_prop(x, w1, b1, w2, b2)
    dw1, db1, dw2, db2 = back_prop(z1,a1,w2,a2, x, y)
    w1, b1, w2, b2, dw1, dw2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, rate)

    if momentum:
        if pdw1 == None:
            pdw1 = 0
            pdw2 = 0
        # print(pdw1)
        w1 = w1 + decay*pdw1
        w2 = w2 + decay*pdw2
       

    return w1, b1, w2, b2, a2, dw1, dw2

def minibatchGD(x, y, epochs, rate, batch_size=128, validation=[],validKey='' ,w1=None, b1=None, w2=None, b2=None, sudoku=None, key=None):
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
        w1, b1, w2, b2 = init_params()
    
    for epoch in range(epochs):
        pos = 0
        dw1 = 0
        dw2 = 0
        while(pos < batch_size):
            batchX = x.T[pos:pos+batch_size].T
            batchY = y.T[pos:pos+batch_size].T
            # display(x.shape, batchX.shape)
            # display(y.shape, batchY.shape)
            pos = pos + batch_size
            dist = int((len(load)*epoch)/epochs)
            bar = ['*']*dist
            loader = [*load]
            loader[0:dist] = bar
            load = ''.join(loader)

            # batch = x.T[pos:epoch*batch_size].T

            w1, b1, w2, b2, a3, dw1, dw2= gradient_descent(x=batchX, y=batchY, rate=rate, momentum=True, decay=.8, w1=w1, b1=b1, w2=w2, b2=b2, dw1=dw1, dw2=dw2)

            accuracy, loss = get_accuracy(get_predictions(a3), batchY)
            xLoss.append(loss)
            yAcc.append(accuracy)

            acc, val_loss = get_accuracy(make_predictions(validation, w1, b1, w2, b2), validKey)
            validationLoss.append(val_loss)
            validationAcc.append(acc)

            if sudoku is not None:
                sudoku_acc, sudoku_loss = get_accuracy(make_predictions(sudoku, w1, b1, w2, b2), key)
                sudokuLoss.append(sudoku_loss)
                sudokuAcc.append(sudoku_acc)



            if epoch < epochs-1:
                print(f'[{epoch}/{epochs}] {load} {accuracy} {loss}', end='\r')
            else:
                print(f'{load} {accuracy} {loss} complete')

        
            # if pos > batch_size

            


    return w1, b1, w2, b2, np.array(xLoss), np.array(yAcc), np.array(validationLoss), np.array(validationAcc), np.array(sudokuLoss), np.array(sudokuAcc)

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


