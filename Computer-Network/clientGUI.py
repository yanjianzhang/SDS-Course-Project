from tkinter import *
from socket import *
from threading import *
from  tkinter  import ttk
from tkinter.scrolledtext import*

#定义客户端的接受函数
class Receive():
  def __init__(self, server, gettext):
    while 1:
      try:
        text = server.recv(1024)
        if not text: break
        #当收到服务器消息的时候就将内容放入下文所设置的文本框
        gettext.configure(state='normal')
        gettext.insert(END,' %s\n'%text.decode())
        gettext.configure(state='disabled')
        gettext.see(END)
      except:
        break
#定义用户段的GUI界面
class App(Thread):
  client = socket(AF_INET,SOCK_DGRAM)
  client.connect(('localhost', 12345))
  def __init__(self, master):
    Thread.__init__(self)
    frame = Frame(master)
    frame.pack()
    # 设置滚动文本框，当客户端发送消息的时候就放入滚动文本框
    self.gettext = ScrolledText(frame, height=20,width=75)
    self.gettext.pack()
    self.gettext.insert(END,'Welcome to Chat\n')
    self.gettext.configure(state='disabled')
    sframe = Frame(frame)
    sframe.pack(anchor='w')
    #设置Input标签
    self.pro = Label(sframe, text="Input>>")
    self.command = StringVar()
    #设置组合下拉框，用于命令的选择和内容的补充输入
    self.textEnter = ttk.Combobox(sframe,width=40,textvariable = self.command)
    self.textEnter['values'] = ("/admin ","/opencorner ","/closecorner ",
                                "/corners","/search ","/enter ","/listusers","/kickout ",
                                "/msg ","/@ private_message ","/back","/leave")
    self.textEnter.bind(sequence="<Return>", func=self.Send)
    self.pro.pack(side=LEFT)
    # self.sendtext.pack(side=LEFT)
    self.textEnter.pack(side=LEFT)
    #设置公开信息按钮
    self.publicmessage = Button(root,text = "Public MS",command = self.publicMessage)
    #设置Only to(只发送到)标签
    self.tag = Label(sframe, text="Only To>>")
    #设置目标用户栏
    self.aimuser = Entry(sframe, width=20)
    self.aimuser.bind(sequence="<Return>", func=self.privateMessage)
    self.aimuser.pack(side=RIGHT)
    self.tag.pack(side=RIGHT)
    #设置私密信息按钮
    self.privatemessage = Button(root, text="Private MS", command=self.privateMessage)
    self.privatemessage.pack(side=RIGHT,ipadx = 10,padx = 20)
    self.publicmessage.pack(side=RIGHT, ipadx=10, padx=10)

    #设置发送私密消息的函数
  def privateMessage(self):
    self.gettext.configure(state='normal')
    message = self.textEnter.get()
    aimuser = self.aimuser.get()
    if message == "": message = " "
    # self.gettext.insert(END, 'Client >> %s\n' % message)
    self.textEnter.delete(0, END)
    self.client.send(("/@"+aimuser+" private_message " + message).encode())
    self.textEnter.focus_set()
    self.gettext.configure(state='disabled')
    self.gettext.see(END)
    return
  #设置发送公开消息的函数
  def publicMessage(self):
    self.gettext.configure(state='normal')
    message = self.textEnter.get()
    if message == "": message = " "
    # self.gettext.insert(END, 'Me >> %s\n' % message)
    self.textEnter.delete(0, END)
    self.client.send(("/msg " + message).encode())
    self.textEnter.focus_set()
    self.gettext.configure(state='disabled')
    self.gettext.see(END)
    return
    #设置发送消息的函数，并实现全自动区分是否是指令、公开消息、私密消息
  def Send(self, args):
    text = self.textEnter.get()
    aimuser = self.aimuser.get()
    #识别是否是指令
    if not text.startswith("/"):
        #判断目标用户栏是否为空
        if aimuser == "":
            self.publicMessage()
        else:
            self.privateMessage()
    else:
        self.gettext.configure(state='normal')
        if text=="": text=" "
        self.gettext.insert(END,'C >> %s\n'%text)
        self.textEnter.delete(0,END)
        self.client.send(text.encode())
        self.textEnter.focus_set()
        self.gettext.configure(state='disabled')
        self.gettext.see(END)
    #设置GUI界面执行的函数
  def run(self):
    Receive(self.client, self.gettext)
root = Tk()
root.title('Client Chat')
app = App(root).start()
root.mainloop()
