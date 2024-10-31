from skimage import io, transform, color
import cv2

import numpy as np


def split_grid(image : np.array):
    m, n = image.shape
    one = image[0:int(m/3),0:int(n/3)]
    two = image[0:int(m/3),int(n/3):int(2*n/3)]
    three = image[0:int(m/3),int(2*n/3):n]

    four = image[int(m/3):int(2*m/3),0:int(n/3)]
    five = image[int(m/3):int(2*m/3),int(n/3):int(2*n/3)]
    six = image[int(m/3):int(2*m/3),int(2*n/3):n]

    seven = image[int(2*m/3):int(m),0:int(n/3)]
    eight = image[int(2*m/3):int(m),int(n/3):int(2*n/3)]
    nine = image[int(2*m/3):int(m),int(2*n/3):n]

    return one, two, three, four, five, six, seven, eight, nine

# squares = list(split_grid(image))
# nine = squares[8]

# spaces = list(split_grid(nine))


def get_pixels(image : np.array, show = False):
    image = transform.resize(image, (28, 28))
    pixels = np.array(image.flatten())
    if show: 
        print(pixels)

    return image, np.array([pixels])


directory = 'assets' 
import os

def create_entry(pixels : np.array, label : str, set : list):
    pixels = pixels.tolist()[0]
    # print(pixels)
    # print(label, type(pixels))
    pixels.extend([label])
    set.append(pixels)
    
    return pixels, set


def add_to_test(test, pixels):
    # print('added')
    return np.append(test, pixels.T, 1)

def format_board(path, test, upload=False):
    if upload:
        path = 'static/uploads/'+path
    # print(path)
    board =  cv2.imread(path)
    # print(board.shape)
    image = cv2.bitwise_not(board)
    image = color.rgb2gray(image)
    image = image.round()

    # io.imshow(image)
    # io.show()

    grids = list(split_grid(image))
    mini_grids = []
    for grid in grids:
        spaces = list(split_grid(grid))
        mini_grids.append(spaces)
        for space in spaces:
            # io.imshow(space)
            # io.show()
            # label = grid_labels[i]
            test = add_to_test(test, get_pixels(space)[1])

    return mini_grids, test, board

def setup(labels):
    grids = {
    '1' : [],
    '2' : [],
    '3' : [],
    '4' : [],
    '5' : [],
    '6' : [],
    '7' : [],
    '8' : [],
    '9' : []
}
    for i in range(0, 9):
        grid = labels[i*9:i*9+9]
        # actual_grid = grid_labels[i*9:i*9+9]

        grid = np.array(grid).reshape(3,3).tolist()
        # actual_grid = np.array(actual_grid).reshape(3,3).tolist()
        grids[str(i+1)] = np.array(grid).reshape(3,3)
        # actual_grids[str(i+1)] = np.array(actual_grid).reshape(3,3)

    return grids

def board_to_text(grids, solving=False): 
    spacer = np.array([["-","-","-","-","-","-","-","-","-","-","-"]])
    spacer2 = np.array([['|'],['|'],['|']])

    layer1 = grids['1']
    if not solving:
        layer1 = np.append(layer1, spacer2, axis=1)
    layer1 = np.append(layer1, grids['2'], axis=1)
    if not solving:
        layer1 = np.append(layer1, spacer2, axis=1)
    layer1 = np.append(layer1, grids['3'], axis=1)
    
    layer2 = grids['4']
    if not solving:
        layer2 = np.append(layer2, spacer2, axis=1)
    # layer2 = np.append(layer2, spacer2, axis=1)
    layer2 = np.append(layer2, grids['5'], axis=1)
    if not solving:
        layer2 = np.append(layer2, spacer2, axis=1)
    # layer2 = np.append(layer2, spacer2, axis=1)
    layer2 = np.append(layer2, grids['6'], axis=1)
    
    layer3 = grids['7']
    if not solving:
        layer3 = np.append(layer3, spacer2, axis=1)
    # layer3 = np.append(layer3,spacer2, axis=1)
    layer3 = np.append(layer3, grids['8'], axis=1)
    if not solving:
        layer3 = np.append(layer3, spacer2, axis=1)
    # layer3 = np.append(layer3, spacer2, axis=1)
    layer3 = np.append(layer3, grids['9'], axis=1)
    
    if not solving:
        layer1 = np.append(layer1, spacer, axis=0)
        layer2 = np.append(layer2, spacer, axis=0)
    
    layout = np.append(np.append(layer1, layer2, axis=0), layer3, axis=0)
    
    return layout


def preds2str(preds):
    pred_grids = setup(preds)
    preds_string = board_to_text(pred_grids, solving=True)
    preds_string = [str(x) for x in preds_string.flatten()]
    return ' '.join(preds_string)


def output_board(numbers):

    spacer = np.zeros((5, 724, 3))
    spacer2 = np.zeros((76,5, 3))


    out_rows = {
    '1' : [],
    '2' : [],
    '3' : [],
    '4' : [],
    '5' : [],
    '6' : [],
    '7' : [],
    '8' : [],   
    '9' : []
}
    
    for i, row in enumerate(numbers):
        output_row = spacer2
        # transform.resize(cv2.imread(f'boards/{numbers[i][0]}.jpeg'), (76,76))
        # out_rows['1'] = output_row
        for j, num in enumerate(row):
            # if j == 0: 
            #     continue
            output_row = np.append(output_row, transform.resize(cv2.imread(f'boards/{num}.jpeg'), (76,76)),  axis=1)
            # print(output_row.shape)
            if (j+1)%3 == 0:
                output_row = np.append(output_row, spacer2, axis=1)

        # print(output_row.shape)
        # io.imshow(output_row)
        # io.show()
        out_rows[str(i+1)] = output_row
            
    
    
    output = spacer
    
    # output = transform.resize(output, (76,760))
    
    # print(output.shape)
    for i in range(1,10): 
        output = np.append(output, transform.resize(out_rows[str(i)], (76,724)), axis=0)
        if i%3 == 0:
            output = np.append(output, spacer, axis=0)
    # io.imshow(output)
    # io.show()
    
    return output
# print(image2.shape)
# io.imshow(image2)
# io.show()