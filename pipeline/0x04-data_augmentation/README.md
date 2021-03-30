This project is about 0x04-data_augmentation
0-flip.py - Write a function def flip_image(image): that flips an image horizontally:
image is a 3D tf.Tensor containing the image to flip
Returns the flipped image

1-crop.py - Write a function def crop_image(image, size): that performs a random crop of an image:
image is a 3D tf.Tensor containing the image to crop
size is a tuple containing the size of the crop
Returns the cropped image

2-rotate.py - Write a function def rotate_image(image): that rotates an image by 90 degrees counter-clockwise:
image is a 3D tf.Tensor containing the image to rotate
Returns the rotated image

3-shear.py - Write a function def shear_image(image, intensity): that randomly shears an image:
image is a 3D tf.Tensor containing the image to shear
intensity is the intensity with which the image should be sheared
Returns the sheared image

4-brightness.py - Write a function def change_brightness(image, max_delta): that randomly changes the brightness of an image:
image is a 3D tf.Tensor containing the image to change
max_delta is the maximum amount the image should be brightened (or darkened)
Returns the altered image

5-hue.py - Write a function def change_hue(image, delta): that changes the hue of an image:
image is a 3D tf.Tensor containing the image to change
delta is the amount the hue should change
Returns the altered image
