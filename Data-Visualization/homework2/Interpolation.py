import math
import numpy as np

def Interpolation(matrix, point):
	x, y = point
	shape_x, shape_y = matrix.shape
	if x < 1 or x > shape_x:
		return None
	if y < 1 or y > shape_y:
		return None
	x1,x2 = math.floor(x)-1, math.ceil(x)-1 # for index usage in matrix
	y1,y2 = math.floor(y)-1, math.ceil(y)-1
	x, y = x-1, y-1
	if x2 == x1:
		f1 = matrix[y1][x1]
		f2 = matrix[y2][x1]
	else:
		f1 = (x2 - x)/ (x2 - x1) * matrix[y1][x1] + (x - x1)/(x2 - x1) * matrix[y1][x2]
		f2 = (x2 - x) / (x2 - x1) * matrix[y2][x1] + (x - x1) / (x2 - x1) * matrix[y2][x2]
	if y2 == y1:
		fp = f1
	else:
		fp = (y2 - y) / (y2 - y1) * f1 + (y - y1)/(y2 - y1) * f2
	return fp


def scale(array, num):
	shape_y, shape_x = array.shape
	new_array = np.zeros((int(shape_x*num), int(shape_y*num)))
	transformed_step = 1/num
	for j in range(int(shape_y*num)):
		for i in range(int(shape_x*num)):
			new_array[j][i] = Interpolation(array,(transformed_step*i, transformed_step*j))
	return new_array


from PIL import Image
im = Image.open("./brain_small.jpg")
image = np.array(im)
H ,W = image.shape
new_image = scale(image, 3)
new_im = Image.fromarray(new_image)
if new_im.mode == 'F':
	new_im = new_im.convert('L')
new_im.save("./scale_brain.jpg")





