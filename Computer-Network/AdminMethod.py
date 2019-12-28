#定义Corner类
class Corner:
    def __init__(self,cornername,language):
        self.cornername = cornername
        self.cornerusers = []
        self.language = language
        self.userid2client = {}
        self.client2userid = {}
#创建一个外语角
def openCorner(cornername,language,addr,s,Corners,Cornername2Conner):
    newCorner = Corner(cornername,language)
    Cornername2Conner[cornername] = newCorner
    Corners.append(newCorner)
    s.sendto((cornername+" has been opened ").encode(), addr)
    return newCorner
#列举外语角
def listCorners(addr,s,Corners):
    if Corners == []:
        s.sendto(("No corner exists").encode(), addr)
    else:
        str =""
        for i in Corners:
            str = str + "Corner: " + i.cornername + "   Lang: " + i.language
            if i != Corners[len(Corners)-1]:
                str = str +"\n "
        s.sendto((str).encode(), addr)
    return
#搜索相应语言的外语角
def searchCorners(addr,s,language,Corners):
    str = ""
    first = 0
    for i in Corners:
        if i.language == language:
            if first == 0:
                str = str + "Corner: " + i.cornername + "   Lang: " + i.language
                first = 1
            else :
                str = str + "\n "+"Corner: " + i.cornername + "   Lang: " + i.language
    s.sendto((str).encode(), addr)
    return
#列举当前外语角的用户
def listUser(addr,s,Client2State):
    if addr in Client2State:
        currentCorner = Client2State[addr]
        str = ""
        for i in currentCorner.cornerusers:
            str = str + i + " "
        s.sendto((str).encode(), addr)
    else:
        s.sendto(("You are not in any corners").encode(), addr)
    return
#踢出用户
def kickoutUser(addr,userid,s,Client2State):
    currentCorner = Client2State[addr]
    if userid in currentCorner.cornerusers:
        for user in currentCorner.cornerusers:
            if user != userid:
                useraddr = currentCorner.userid2client[user]
                s.sendto((userid + " has been removed from "+ currentCorner.cornername).encode(), useraddr)
            else:
                useraddr = currentCorner.userid2client[user]
                s.sendto(("You have been removed from " + currentCorner.cornername +" by Admin").encode(), useraddr)
        currentCorner.cornerusers.remove(userid)
        del currentCorner.userid2client[userid]
    else:
        s.sendto((userid + " is not in current corner " + currentCorner.cornername).encode(), addr)
    return
#进入外语角
def enterCorner(Cornername,addr,username,s,Cornername2Conner,Client2State):
    if Cornername in Cornername2Conner:
        currentCorner = Cornername2Conner[Cornername]
        if username in currentCorner.userid2client:
            oldaddr = currentCorner.userid2client[username]
            olduserid = currentCorner.client2userid[oldaddr]
            currentCorner.cornerusers.remove(olduserid)
            del currentCorner.client2userid[oldaddr]
            # del currentCorner.userid2client[userid]
            s.sendto(("Leave from "+currentCorner.cornername +" because you logined " +  "in the other place").encode(), oldaddr)
        userid = username
        currentCorner.cornerusers.append(userid)
        currentCorner.client2userid[addr] = userid
        currentCorner.userid2client[userid] = addr
        Client2State[addr] = currentCorner
        s.sendto((userid + " has logined " + currentCorner.cornername).encode(), addr)
    else: s.sendto(("No such corner: " + Cornername).encode(), addr)
    return
#关闭外语角
def closeCorner(cornername,addr,s,Corners,Cornername2Conner,Client2State):
    if cornername in Cornername2Conner:
        currentCorner = Cornername2Conner[cornername]
        for user in currentCorner.cornerusers:
            useraddr = currentCorner.userid2client[user]
            s.sendto((cornername + " has been closed, You have been removed").encode(), useraddr)
            del Client2State[useraddr]
        Corners.remove(Cornername2Conner[cornername])
        del Cornername2Conner[cornername]
        s.sendto((cornername + " has been closed").encode(), addr)
    else:
        s.sendto((cornername + " does not exist").encode(), addr)
    return
#退出外语角
def back(addr,s,Client2State):
    if addr in Client2State:
        currentCorner = Client2State[addr]
        userid = currentCorner.client2userid[addr]
        currentCorner.cornerusers.remove(userid)
        del currentCorner.client2userid[addr]
        del Client2State[addr]
        # del currentCorner.userid2client[userid]
        s.sendto(("You leave "+currentCorner.cornername+" corner").encode(), addr)
    else:
        s.sendto(("You are not in any corners").encode(), addr)
#退出程序
def leave(addr,s,Client2State):
    if addr in Client2State:
        currentCorner = Client2State[addr]
        userid = currentCorner.client2userid[addr]
        currentCorner.cornerusers.remove(userid)
        del currentCorner.client2userid[addr]
        del Client2State[addr]
        # del currentCorner.userid2client[userid]
    s.sendto(("You leave system").encode(), addr)
    return
#发送私密消息
def privateMessage(addr,s,Client2State,userid2,message):
    if addr in Client2State:
        currentCorner = Client2State[addr]
        userid = currentCorner.client2userid[addr]
        if userid2 in currentCorner.cornerusers:
            s.sendto((userid+"@"+userid2+": " +message).encode(), currentCorner.userid2client[userid2])
            s.sendto((userid + "@" + userid2 + ": " + message).encode(), addr)
        else:
            s.sendto((userid2 + " is not in current corner").encode(), addr)
    else:
        s.sendto(("You are not in any corners").encode(), addr)
    return
#发送公开消息
def publicMessage(addr,s,Client2State,message):
    if addr in Client2State:
        currentCorner = Client2State[addr]
        userid = currentCorner.client2userid[addr]
        for user in currentCorner.cornerusers:
            useraddr = currentCorner.userid2client[user]
            s.sendto((userid + ": "+message).encode(), useraddr)
    else:
        s.sendto(("You are not in any corners").encode(), addr)
    return