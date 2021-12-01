from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title
from pylab import ginput,close


def point_extract_interative(image1):
    colors = "bgrcmykw"
    # 展示图片
    im1 = array(Image.open(image1))
    imshow(im1)
    # 初始化组别数
    group_no = int(input("Please input the number of groups:"))
    group_num = input("Please input the size of each group:").split()
    group_num = [int(v) for v in group_num]
    dic = {}
    #读每组标记点
    for i in range(group_no):
        #提醒用户信息
        print('Please click '+ str(group_num[i])+' points for group '+ str(i)+":")
        #获取标记点
        person = ginput(group_num[i])
        #可视化标记结果
        x = [x for x, y in person]
        y = [y for x, y in person]
        print('you clicked:', person)
        title('Points: group '+str(group_num[i]))
        plot(x, y, '*', c=colors[i])
        dic[i] = person
    print(dic[0])
    close()
    return group_num, dic


if __name__ == '__main__':
    colors = "bgrcmykw"
    _, person = point_extract_interative('photo.jpg')
    _, animal = point_extract_interative('ape.png')
    print(person)
    print(animal)