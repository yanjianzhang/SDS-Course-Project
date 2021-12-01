from skimage import exposure
from skimage import data
from skimage import io, data, img_as_float
# import Image as Img
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import copy
from skimage.color import rgb2gray
from scipy.fftpack import fft2, ifft2



def low_pass(image):

	M, N = image.shape
	P,Q = 2*M, 2*N
	pad_image = np.ones((P, Q))
	for i in range(M):
		for j in range(N):
			pad_image[i][j] = image[i][j]

	fc = np.zeros((P, Q))
	for i in range(P):
		for j in range(Q):
			fc[i][j] = pad_image[i][j] * ((-1) ** (i + j))

	f_image = fft2(fc)  #  2-D discrete Fourier transform.



	plt.subplot(2, 2, 2)
	plt.imshow(np.log(np.abs(f_image)), "gray")
	plt.axis('off')
	plt.title('Fourier transform')


	low_pass_filter = np.zeros((P, Q))
	size = 100
	for i in range(P):
		for j in range(Q):
			if (i-M)**2+(j-N)**2 < size**2:
				low_pass_filter[i][j] = 1
	# low_pass_filter[M-size: M+size, N-size: N+size] = np.ones((size*2, size*2))
	new_f_image = f_image*low_pass_filter

	plt.subplot(2, 2, 3)
	plt.imshow(np.log(abs(new_f_image)), "gray")
	plt.title('Low Pass')

	re_image = ifft2(new_f_image)

	fc1 = np.zeros((P, Q))
	for i in range(P):
		for j in range(Q):
			fc1[i][j] = re_image[i][j] * ((-1) ** (i + j))
	re_image = fc1[:M,:N]

	plt.subplot(2, 2, 4)
	plt.imshow(re_image, "gray")
	plt.title('Reverse Fourier')


def notch_filter(image):

	M, N = image.shape
	P,Q = 2*M, 2*N
	pad_image = np.ones((P, Q))
	for i in range(M):
		for j in range(N):
			pad_image[i][j] = image[i][j]

	fc = np.zeros((P, Q))
	for i in range(P):
		for j in range(Q):
			fc[i][j] = pad_image[i][j] * ((-1) ** (i + j))

	f_image = fft2(fc)  #  2-D discrete Fourier transform.
	plt.subplot(2, 2, 2)
	plt.imshow(np.log(np.abs(f_image)), "gray")
	plt.axis('off')
	plt.title('Fourier transform')
	notch_filt = np.ones((P, Q))

	abs_f_image = abs(f_image)
	size = 30
	center = 60
	for i in range(size,P-size):
		for j in range(size,Q-size):
			if not (i-M)**2+(j-N)**2 < center**2 and f_image[i][j] > 1.8 * np.mean(abs_f_image[i-size:i+size, j-size:j +size]): # keep the center part
				notch_filt[i-1:i+2,j-1:j+2] = np.zeros((3,3))

	new_f_image = f_image * notch_filt

	plt.subplot(2, 2, 3)
	plt.imshow(np.log(abs(new_f_image)), "gray")
	plt.title('Notch Filter')

	re_image = ifft2(new_f_image)

	fc1 = np.zeros((P, Q))
	for i in range(P):
		for j in range(Q):
			fc1[i][j] = re_image[i][j] * ((-1) ** (i + j))
	re_image = fc1[:M,:N]

	plt.subplot(2, 2, 4)
	plt.imshow(re_image, "gray")
	plt.title('Reverse Fourier')


# low-freq filter
image = io.imread("./image/slice.jpg")
dtype = image.dtype.type
print(image)
image =  rgb2gray(image)
image = dtype(list([int(i*255) for i in j] for j in image))
io.imsave("./image/pattern_grey.jpg", image)
plt.subplot(2, 2, 1)
plt.imshow(image, plt.cm.gray, label='First Line')
plt.axis('off')
plt.title('Origin Image')
image = np.array(image)
low_pass(image)
plt.show()

# notch filter
image = io.imread("./image/freq_testimage_shepplogan.PNG")
dtype = image.dtype.type
image =  rgb2gray(image)
image = dtype(list([int(i*255) for i in j] for j in image))
io.imsave("./image/pattern_grey.jpg", image)
plt.subplot(2, 2, 1)
plt.imshow(image, "gray")
plt.axis('off')
plt.title('Origin Image')
notch_filter(image)
plt.show()