from three_hidden import *
# from fourLayer import *
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
            # print(label, type(pixels))
            pixels.extend([label])
            # print(pixels)
            # pixels = np.append(pixels, ), 1)

            set.append(pixels)
            
            return pixels, set
        
        digits = []
        print('loading files...')
        for folder in os.listdir(directory):
            f = os.path.join(directory, folder)

            # print(f)
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
        digits = digits.round()
        np.random.shuffle(digits)
        # one = digits[0:2]
        m, n = digits.shape

        
        digit_train = digits.T
        digit_labels = digit_train[-1]
        x_digit_train = digit_train[0:n-1]
        first_digit = x_digit_train[:,1]
        first = x_train[:,1]
        # x_digit_train = min_max_normalize(digit_train[0:n-1])
        print(first_digit[first_digit != 0], first[first != 0], first_digit.shape == first.shape)
        io.imshow(first.reshape(28,28))
        io.show()
        io.imshow(first_digit.reshape(28,28))
        io.show()
        # first.reshape(28)
        # digit_test = digits[int(2*m/3):m].T
        # test_labels = digit_test[-1]
        # x_digit_test = digit_test[0:n-1]

        test = np.array(pd.read_csv('test.csv')).T

        image = cv2.imread('number.png')

        # digit_test = np.array([])
        # x_digit_test = min_max_normalize(digit_test[0:n-1])
        # train = digits[0:783]
        # print(label)
        # print(digit_labels[0:50])

        test = min_max_normalize(test)


        def add_to_test(test, pixels):
            # print('added')
            return np.append(test, pixels.T, 1)
        
        grid_labels = np.array([0,7,0,0,0,0,0,3,0,
                                8,3,0,4,0,0,0,0,0,
                                0,4,1,0,0,7,0,0,0,
                                0,4,0,0,2,1,0,0,5,
                                7,0,0,3,0,0,0,0,0,
                                0,6,0,4,0,8,0,0,0,
                                9,0,0,0,0,8,0,0,0,
                                0,0,5,0,0,0,0,0,0,
                                2,8,0,0,0,0,1,0,9])
        print(len(grid_labels))
        def format_board(test):
            image = cv2.imread('board.png')
            image = cv2.bitwise_not(image)
   
            image = color.rgb2gray(image)

            io.imshow(image)
            io.show()

            # i = 0
            grids = list(split_grid(image))
            mini_grids = []
            for grid in grids:
                spaces = list(split_grid(grid))
                mini_grids.append(spaces)
                for space in spaces:
                    # io.imshow(space)
                    # io.show()
                    # label = grid_labels[i]
                    # i+=1
                    test = add_to_test(test, get_pixels(space)[1])
                    # entry, digits = create_entry(get_pixels(space)[1], label, digits)

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

        print(board_squares[:,1][board_squares[:,1] != 1], len(board_squares[:,0]))

        rate = .1
        iters = 1300

        w1, b1, w2, b2 = gradient_descent(x_train, labels_train, iters, rate)
        # # print(w1, b1, w2, b2, w3, b3)


        # preds = make_predictions(x_test, w1, b1, w2, b2, w3, b3)
        # print(preds, labels_test)
        # grid_accuracy = get_accuracy(preds, labels_test)
        # print(f'test accuracy : {grid_accuracy}')



        w1, b1, w2, b2= gradient_descent(x_digit_train, digit_labels.astype(int), 1300, .12, w1, b1, w2, b2)
        # # print(w1, b1, w2, b2, w3, b3)



# for printed digits
        # preds = make_predictions(x_digit_test, w1, b1, w2, b2, w3, b3)
        # print(preds, test_labels)
        # print(f'test accuracy : {get_accuracy(preds, test_labels.astype(int))}')


       

        # print(test)
        # print(test[-1])
        # np.append(labels_test, 2, 0)

        # print(test[:,0])
        preds = make_predictions(board_squares, w1, b1, w2, b2)
        print(f'{preds} \n {grid_labels}')
        print(f'test accuracy : {get_accuracy(preds, grid_labels)}')
        indexes = [j + 1 for j in range(preds.shape[0])]

        preds = zip(indexes, preds)
        submission = pd.DataFrame(preds, columns = ['ImageID', 'Label'])

        submission.to_csv('puzzle.csv', index=False)

main()