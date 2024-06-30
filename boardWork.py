from skimage import io, transform, color
import cv2

import numpy as np

image = cv2.imread('board.png')
# if image.shape[2] == 4:
#     image = color.rgb2gray(color.rgba2rgb(image))
# else: 
#     image = color.rgb2gray(image)
image = cv2.bitwise_not(image)
# print(np.array(image).shape)

# image = transform.resize(image, (28, 28))

image = color.rgb2gray(image)



#   ---|---|---
#   


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

squares = list(split_grid(image))
nine = squares[8]

spaces = list(split_grid(nine))


def get_pixels(image : np.array):
    # imm, imn = image.shape
    # print(image)
    # buffer = np.zeros((imm + 150, imn + 150))
    # m, n = buffer.shape
    # image_zoom = image[75:imm-75][75:imn-75]
    # print(image_zoom)
    # buffer[75:m-75,75:n-75] = image

    image = transform.resize(image, (28, 28))
    pixels = np.array(image.flatten())

    return image, np.array([pixels])



# print(image)
# print(one)
print()


# print(image2.shape)
# io.imshow(image2)
# io.show()