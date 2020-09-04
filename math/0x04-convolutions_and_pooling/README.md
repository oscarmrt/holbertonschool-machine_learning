This project is about 0x04. Convolutions and Pooling
0-convolve_grayscale_valid.py - Write a function def convolve_grayscale_valid(images, kernel): that performs a valid convolution on grayscale images:
images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
kh is the height of the kernel
kw is the width of the kernel
You are only allowed to use two for loops; any other loops of any kind are not allowed
Returns: a numpy.ndarray containing the convolved images

1-convolve_grayscale_same.py - Write a function def convolve_grayscale_same(images, kernel): that performs a same convolution on grayscale images:
images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
kh is the height of the kernel
kw is the width of the kernel
if necessary, the image should be padded with 0’s
You are only allowed to use two for loops; any other loops of any kind are not allowed
Returns: a numpy.ndarray containing the convolved images

2-convolve_grayscale_padding.py - Write a function def convolve_grayscale_padding(images, kernel, padding): that performs a convolution on grayscale images with custom padding:
images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
kh is the height of the kernel
kw is the width of the kernel
padding is a tuple of (ph, pw)
ph is the padding for the height of the image
pw is the padding for the width of the image
the image should be padded with 0’s
You are only allowed to use two for loops; any other loops of any kind are not allowed
Returns: a numpy.ndarray containing the convolved images

3-convolve_grayscale.py - Write a function def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on grayscale images:
images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
kh is the height of the kernel
kw is the width of the kernel
padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
if ‘same’, performs a same convolution
if ‘valid’, performs a valid convolution
if a tuple:
ph is the padding for the height of the image
pw is the padding for the width of the image
the image should be padded with 0’s
stride is a tuple of (sh, sw)
sh is the stride for the height of the image
sw is the stride for the width of the image
You are only allowed to use two for loops; any other loops of any kind are not allowed Hint: loop over i and j
Returns: a numpy.ndarray containing the convolved images

4-convolve_channels.py - Write a function def convolve_channels(images, kernel, padding='same', stride=(1, 1)): that performs a convolution on images with channels:
images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
c is the number of channels in the image
kernel is a numpy.ndarray with shape (kh, kw, c) containing the kernel for the convolution
kh is the height of the kernel
kw is the width of the kernel
padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
if ‘same’, performs a same convolution
if ‘valid’, performs a valid convolution
if a tuple:
ph is the padding for the height of the image
pw is the padding for the width of the image
the image should be padded with 0’s
stride is a tuple of (sh, sw)
sh is the stride for the height of the image
sw is the stride for the width of the image
You are only allowed to use two for loops; any other loops of any kind are not allowed
Returns: a numpy.ndarray containing the convolved images

5-convolve.py - Write a function def convolve(images, kernels, padding='same', stride=(1, 1)): that performs a convolution on images using multiple kernels:
images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
c is the number of channels in the image
kernels is a numpy.ndarray with shape (kh, kw, c, nc) containing the kernels for the convolution
kh is the height of a kernel
kw is the width of a kernel
nc is the number of kernels
padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
if ‘same’, performs a same convolution
if ‘valid’, performs a valid convolution
if a tuple:
ph is the padding for the height of the image
pw is the padding for the width of the image
the image should be padded with 0’s
stride is a tuple of (sh, sw)
sh is the stride for the height of the image
sw is the stride for the width of the image
You are only allowed to use three for loops; any other loops of any kind are not allowed
Returns: a numpy.ndarray containing the convolved images

6-pool.py - Write a function def pool(images, kernel_shape, stride, mode='max'): that performs pooling on images:
images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
c is the number of channels in the image
kernel_shape is a tuple of (kh, kw) containing the kernel shape for the pooling
kh is the height of the kernel
kw is the width of the kernel
stride is a tuple of (sh, sw)
sh is the stride for the height of the image
sw is the stride for the width of the image
mode indicates the type of pooling
max indicates max pooling
avg indicates average pooling
You are only allowed to use two for loops; any other loops of any kind are not allowed
Returns: a numpy.ndarray containing the pooled images
