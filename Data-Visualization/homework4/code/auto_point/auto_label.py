from tkinter import *
from PIL import Image, ImageTk
import os
import os
import glob
import random
import pickle
from pylab import array, imshow, title, show
from locally_affine import  locally_affine
import numpy as  np
from interpolation import linearInter

IMAGE_SIZE = 200, 200

COLORS = ['red', 'orange',   'green', 'blue', 'indigo', 'violet', 'gray',  'black']

class Panel():
	def __init__(self, root, default_path):
		root.title("Point Tool")
		self.frame = Frame()
		self.frame.pack(fill=BOTH, expand=1)
		self.frame.columnconfigure(1, weight=1)
		self.frame.rowconfigure(4, weight=1)

		# 图片路径输入框，可以是相对路径也可以是绝对路径
		self.label = Label(self.frame, text="Image Path")
		self.label.grid(row=0, column = 1, sticky= E)
		self.entry = Entry(self.frame)
		self.entry.insert(END, default_path)
		self.entry.grid(row=0, column = 1, sticky= W + E)

		# 加载按钮
		self.loadButton = Button(self.frame, text="Load", command = self.loadImage)
		self.loadButton.grid(row=0, column = 2, sticky= W + E)

		# 定义展示幕布
		self.canvas = Canvas(self.frame, cursor='tcross')
		self.canvas.grid(row=1, column=1, rowspan=4, sticky=W + N)
		self.canvas.bind("<Button-1>", self.mouseClick)
		self.canvas.bind("<Motion>", self.mouseMove)

		# 定义点列表
		self.points_label = Label(self.frame, text = 'Points')
		self.points_label.grid(row=1, column=2, sticky=W + N)
		self.listbox = Listbox(self.frame, width=28, height=12)
		self.listbox.grid(row=2, column=2, sticky=N)

		self.save_botton = Button(self.frame, text='Save', command=self.save)
		self.save_botton.grid(row=3, column=2, sticky=W + E + N)

		self.points = []


		# 初始化鼠标状态
		self.STATE = {}
		self.STATE['click'] = 0
		self.STATE['x'], self.STATE['y'] = 0, 0

		self.disp = Label(self.frame, text='')
		self.hl, self.vl, self.current_image = None, None, None


		self.image_w, self.image_h = None, None

	def loadImage(self):
		image = Image.open(self.entry.get())
		self.image_w, self.image_h = image.size
		image = image.resize(
			(IMAGE_SIZE[0], IMAGE_SIZE[1]), Image.ANTIALIAS)
		self.current_image = ImageTk.PhotoImage(image)
		self.canvas.config(width=max(self.current_image.width(), IMAGE_SIZE[0]),
		                      height=max(self.current_image.height(), IMAGE_SIZE[1]))
		self.canvas.create_image(0, 0, image=self.current_image, anchor=NW)

		file_name = self.entry.get().split("/")[-1].split(".")[0]
		if os.path.exists("../labels/" + file_name + ".txt"):
			with open("../labels/" + file_name + ".txt", "rb") as file:
				self.points = pickle.load(file)
				for x, y, idx in [(self.points[i][0], self.points[i][1], i)  for i in range(len(self.points))]:

					self.listbox.insert(END, '(%.2f, %.2f)' % (x, y))
					self.listbox.itemconfig(
						idx, fg=COLORS[idx % len(COLORS)])

					x1 = x * IMAGE_SIZE[0] / self.image_w
					y1 = y * IMAGE_SIZE[1] / self.image_h

					self.drawCircle(x1, y1, 2, fill=COLORS[idx % len(COLORS)])

	def mouseClick(self, event):

		self.STATE['x'], self.STATE['y'] = event.x, event.y
		x1 = self.STATE['x']
		y1 = self.STATE['y']

		x1 = x1 / IMAGE_SIZE[0] * self.image_w
		y1 = y1 / IMAGE_SIZE[1] * self.image_h

		self.points.append((x1, y1))
		self.listbox.insert(END, '(%.2f, %.2f)' % (x1, y1))
		self.listbox.itemconfig(
			len(self.points) - 1, fg=COLORS[(len(self.points) - 1) % len(COLORS)])
		self.drawCircle( self.STATE['x'], self.STATE['y'], 2, fill=COLORS[(len(self.points) - 1) % len(
			COLORS)])
		self.STATE['click'] = 1 - self.STATE['click']

	def mouseMove(self, event):
		self.disp.config(text='x: %.2f, y: %.2f' % (
			event.x / IMAGE_SIZE[0], event.y / IMAGE_SIZE[1]))  # 鼠标移动时显示当前位置的坐标

		# 如果有图像的话，当移动鼠标时，展示十字线用来定位
		if self.current_image:
			if self.hl:
				self.canvas.delete(self.hl)
			self.hl = self.canvas.create_line(
				0, event.y, self.current_image.width(), event.y, width=2)
			if self.vl:
				self.canvas.delete(self.vl)
			self.vl = self.canvas.create_line(
				event.x, 0, event.x, self.current_image.height(), width=2)

	def drawCircle(self, x, y, r, **kwargs):
		return self.canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)


	def save(self):
		file_name = self.entry.get().split("/")[-1].split(".")[0]
		with open("../labels/"+ file_name +".txt", "wb") as file:
			pickle.dump(self.points, file)



if __name__ == '__main__':
	root = Tk()
	tool = Panel(root = root, default_path ='../images/ape.png')
	tool2 = Panel(root = root, default_path='../images/photo.jpg')

	root.mainloop()


