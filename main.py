# from model import *
from twoLayer import *
# from PIL import Image
# from skimage import io, color, util, transform

def main():
    if __name__ == '__main__':
        rate = .5
        iters = 1250
        w1, b1, w2, b2, w3, b3 = gradient_descent(x_train, labels_train, iters, rate)
        
        # print(w1, b1, w2, b2, w3, b3)
        preds = make_predictions(x_test, w1, b1, w2, b2, w3, b3)
        print(preds, labels_test)
        print(f'test accuracy : {get_accuracy(preds, labels_test)}')


        test = np.array(pd.read_csv('test.csv')).T

        image = io.imread('image.png')
        if image.shape[2] == 4:
            image = color.rgb2gray(color.rgba2rgb(image))
        else: 
            image = color.rgb2gray(image)
        # image = image.reshape(28,28)
        # print(np.array(image).shape)

        image = transform.resize(image, (28, 28))
        pixels = np.array([image.flatten()])

        # print(test[0], test.shape, '\n')
        # print(pixels[0], pixels.shape, '\n')

        test = min_max_normalize(test)
        test = np.append(test, pixels.T, 1)

        # print(test)
        # print(test[-1])
        # np.append(labels_test, 2, 0)

        # print(test[:,0])
        preds = make_predictions(test, w1, b1, w2, b2, w3, b3)
        indexes = [j + 1 for j in range(preds.shape[0])]

        preds = zip(indexes, preds)
        submission = pd.DataFrame(preds, columns = ['ImageID', 'Label'])

        print(submission)

main()