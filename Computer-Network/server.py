import socket
import AdminMethod
import ClientMethod

#有关的全局变量
BUFSIZE = 1024
# clients = []
Corners = []
Cornername2Conner = {}
Adminaddr = []
Client2State = {}
Serveraddr = "127.0.0.1"

#指令处理函数
def Deal(sock,addr,data):
        parse = data.split(" ")
        #在此处设置一个是否有服务器返回消息的条件变量，如果没有调用任何服务器函数，就向客户端说明指令异常
        hasRetValue = 0
        #对管理员身份进行验证
        if parse[0] == "/admin" and len(parse) == 2 and parse[1] == "pw":
            s.sendto(("Admin login success").encode(), addr)
            Adminaddr.append(addr)
            return
        #如果相应的地址是包含在管理员的数组里面的，那么就调用AdminMethod中的函数，按管理员的模式来处理指令
        if addr in Adminaddr:
            if parse[0] == '/opencorner' and len(parse) == 3:
                AdminMethod.openCorner(parse[1],parse[2],addr,sock,Corners,Cornername2Conner)
                hasRetValue = 1
            if parse[0] == '/corners' and len(parse) == 1:
                AdminMethod.listCorners(addr,sock,Corners)
                hasRetValue = 1
            if parse[0] == '/search' and len(parse) == 2:
                AdminMethod.searchCorners(addr,sock,parse[1],Corners)
                hasRetValue = 1
            if parse[0] == '/enter' and len(parse) == 2:
                AdminMethod.enterCorner(parse[1],addr,"admin",sock,Cornername2Conner,Client2State)
                hasRetValue = 1
            if parse[0] == '/listusers' and len(parse) == 1:
                AdminMethod.listUser(addr,sock,Client2State)
                hasRetValue = 1
            if parse[0] == '/back' and len(parse) == 1:
                AdminMethod.back(addr,sock,Client2State)
                hasRetValue = 1
            if parse[0] == '/leave' and len(parse) == 1:
                AdminMethod.leave(addr,sock,Client2State)
                hasRetValue = 1
                # clients.remove(addr)
                print('%s:%s logout' % addr)
            if parse[0] == '/closecorner' and len(parse) == 2:
                AdminMethod.closeCorner(parse[1],addr,sock,Corners,Cornername2Conner,Client2State)
                hasRetValue = 1
            if parse[0].startswith("/@") and parse[1] == "private_message":
                AdminMethod.privateMessage(addr, sock, Client2State, parse[0][2:], data[17+len(parse[0]):])
                hasRetValue = 1
            if parse[0] == "/msg":
                AdminMethod.publicMessage(addr, sock, Client2State, data[5:])
                hasRetValue = 1
            if parse[0] == '/kickout' and len(parse) == 2:
                AdminMethod.kickoutUser(addr,parse[1],sock,Client2State)
                hasRetValue = 1
            #如果没有调用任何服务器函数，就向客户端说明指令异常
            if hasRetValue == 0:
                s.sendto(("Nothing happen, check the wrong of input").encode(), addr)
            return
        # 如果相应的地址是不包含在管理员的数组里面的，那么就调用ClientMethod中的函数，按普通用户的模式来处理指令
        if parse[0] == '/corners' and len(parse) == 1:
            ClientMethod.listCorners(addr,sock,Corners)
            hasRetValue = 1
        if parse[0] == '/enter' and len(parse) == 3:
            ClientMethod.enterCorner(parse[1],addr,parse[2],sock,Cornername2Conner,Client2State)
            hasRetValue = 1
        if parse[0] == '/listusers' and len(parse) == 1:
            ClientMethod.listUser(addr,sock,Client2State)
            hasRetValue = 1
        if parse[0] == '/search' and len(parse) == 2:
            AdminMethod.searchCorners(addr, sock, parse[1], Corners)
            hasRetValue = 1
        if parse[0] == '/back' and len(parse) == 1:
            ClientMethod.back(addr,sock,Client2State)
            hasRetValue = 1
        if parse[0] == '/leave' and len(parse) == 1:
            ClientMethod.leave(addr,sock,Client2State)
            hasRetValue = 1
            # clients.remove(addr)
            print('%s:%s logout' % addr)
        if parse[0].startswith("/@") and parse[1] == "private_message":
            ClientMethod.privateMessage(addr,sock,Client2State,parse[0][2:],data[24:])
            hasRetValue = 1
        if parse[0] == "/msg":
            ClientMethod.publicMessage(addr,sock,Client2State,data[5:])
            hasRetValue = 1
        # 如果没有调用任何服务器函数，就向客户端说明指令异常
        if hasRetValue == 0:
            s.sendto(("Nothing happen, check the wrong of input").encode(), addr)

# 创建一个套接字UDPsocket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 将链接进行绑定
s.bind((Serveraddr, 12345))

#进行无限循环，接收客户端消息
while True:
    print('Bound UDP on 12345……')
    print('waiting for connection...')
    data, addr = s.recvfrom(BUFSIZE)
    print('Received from %s:%s.'%addr)
    currentclient = addr
    Deal(s,addr,data.decode())


