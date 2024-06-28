# from model import *
from twoLayer import *
def main():
    if __name__ == '__main__':
        rate = .5
        iters = 1250
        w1, b1, w2, b2, w3, b3 = gradient_descent(x_train, labels_train, iters, rate)
        

        preds = make_predictions(x_test, w1, b1, w2, b2, w3, b3)
        print(f'test accuracy : {get_accuracy(preds, labels_test)} \n Learning Rate: {rate} Iterations : {iters}')

main()