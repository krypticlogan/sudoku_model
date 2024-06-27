from model import *
def main():
    if __name__ == '__main__':
        w1, b1, w2, b2 = gradient_descent(x_train, labels_train, 1500, .12)
        

        preds = make_predictions(x_test, w1, b1, w2, b2)
        print(f'test accuracy : {get_accuracy(preds, labels_test)}')

main()