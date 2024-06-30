# from model import *
from fourLayer import *
from boardWork import split_grid, get_pixels
# from PIL import Image
from skimage import io, color, util, transform

import os


import cv2
def main():
    if __name__ == '__main__':

        directory = 'assets'

        
        # print(picture)
        # io.imshow(picture)
        # io.show()

        def create_entry(pixels : np.array, label : str, set : list):
            pixels = pixels.tolist()[0]
            # print(pixels)
            print(label, type(pixels))
            pixels.extend([label])
            # print(pixels)
            # pixels = np.append(pixels, ), 1)

            set.append(pixels)
            
            return pixels, set
        
        digits = []
        for folder in os.listdir(directory):
            f = os.path.join(directory, folder)

            print(f)
            for file in os.listdir(f):
                picture = None
                file_name = os.path.join(f, file)
                # print(file)
                if os.path.isfile(file_name):
                    # print(file_name)
                    picture = cv2.imread(file_name)
                    picture = color.rgb2gray(picture)

                    img, pixels = get_pixels(picture)
                    pixels = pixels
                    label = int(file_name[7])
                    entry, digits = create_entry(pixels, label, digits)

                        # print(digits)
                        # io.imshow()
                        # io.show()
                        # print(label, len(entry))
                
        digits = np.array(digits)
        np.random.shuffle(digits)
        # one = digits[0:2]
        m, n = digits.shape
        digit_train = digits[0:int(2*m/3)].T
        digit_labels = digit_train[-1]
        x_digit_train = digit_train[0:n-1]


        digit_test = digits[int(2*m/3):m].T
        test_labels = digit_test[-1]
        x_digit_test = digit_test[0:n-1]
        # train = digits[0:783]
        # print(label)
        # print(digit_labels[0:50])

        rate = .1
        iters = 10000
        w1, b1, w2, b2, w3, b3 = gradient_descent(x_digit_train, digit_labels.astype(int), iters, rate)
        # # print(w1, b1, w2, b2, w3, b3)


        preds = make_predictions(x_digit_test, w1, b1, w2, b2, w3, b3)
        print(preds, digit_labels)
        print(f'test accuracy : {get_accuracy(preds, test_labels.astype(int))}')


        test = np.array(pd.read_csv('test.csv')).T

        image = cv2.imread('number.png')
        
        # image = image.reshape(28,28)
        # print(np.array(image).shape)
        # print(type(image))

# rework this for opencv

        image = color.rgb2gray(image)
        image = transform.resize(image, (28, 28))
        pixels = np.array([image.flatten()])

        




        # print(test[0], test.shape, '\n')
        # print(pixels[0], pixels.shape, '\n')

        test = min_max_normalize(test)

        def add_to_test(test, pixels):
            # print('added')
            return np.append(test, pixels.T, 1)
        
        def format_board(test):
            image = cv2.imread('board.png')
            image = cv2.bitwise_not(image)
   
            image = color.rgb2gray(image)

            io.imshow(image)
            io.show()
            grids = list(split_grid(image))
            mini_grids = []
            for grid in grids:
                spaces = list(split_grid(grid))
                mini_grids.append(spaces)
                for space in spaces:
                    # io.imshow(space)
                    # io.show()
                    test = add_to_test(test, get_pixels(space)[1])

            return mini_grids, test

        # image2, pixels2 = get_pixels(spaces[1])

        # test = np.append(test, pixels.T, 1) # should be 2
        # test = np.append(test, pixels2.T, 1) # should be 8


        print('test size' , test.shape)
        grids, test = format_board(test)
        # print(grids)
        print('test size' , test.shape)
        m, n = test.shape
        board_squares = test[:, 28000:n]
        print(board_squares.shape)
        print()

        # print(test)
        # print(test[-1])
        # np.append(labels_test, 2, 0)

        # print(test[:,0])
        preds = make_predictions(board_squares, w1, b1, w2, b2, w3, b3)
        indexes = [j + 1 for j in range(preds.shape[0])]

        preds = zip(indexes, preds)
        submission = pd.DataFrame(preds, columns = ['ImageID', 'Label'])

        submission.to_csv('puzzle.csv', index=False)

main()