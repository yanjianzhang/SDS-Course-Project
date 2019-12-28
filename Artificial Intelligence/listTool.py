from math import exp

# def multiply(x,y):
# 	if len(x)


def mydot(x,y):
	# print(type(x), type(y))
	# print(isinstance(x, list), isinstance(y, list), len(x), len(y))
	if isinstance(x,list) and isinstance(y,list):
		if len(x[0]) != len(y):
			print("length not equal, multiply")
		return [sum([x[i][j]*y[j] for j in range(len(x[i]))]) for i in range(len(x))]

	if isinstance(x, float) and isinstance(y, list):
		return [x*y[i] for i in range(len(y))]

	if isinstance(x, list) and isinstance(y, float):
		return [x[i]*y for i in range(len(x))]

def minus(x,y):
	if isinstance(x,list) and isinstance(y,list):
		if len(x) != len(y):
			print("length not equal, minus")
		return [x[i]-y[i] for i in range(len(x))]

	if isinstance(x, float) and isinstance(y, list):
		return [x-y[i] for i in range(len(y))]

	if isinstance(x, list) and isinstance(y, float):
		return [x[i]-y for i in range(len(x))]


def add(x,y):
	if isinstance(x,list) and isinstance(y,list):
		if len(x) != len(y):
			print("length not equal, multiply")
		return [x[i]+y[i] for i in range(len(x))]

	if isinstance(x, float) and isinstance(y, list):
		return [x+y[i] for i in range(len(y))]

	if isinstance(x, list) and isinstance(y, float):
		return [x[i]+y for i in range(len(x))]

def list_fn(x, fn):
	return [fn(item) for item in x]

def list_sigmoid(x):
	return [sigmoid(item) for item in x]

def sigmoid(value):
	return 1.0 / (1 + exp(-value))


