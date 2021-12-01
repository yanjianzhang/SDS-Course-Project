from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title
from pylab import show


def point_extract():

    # 绿，红，蓝，鼻子 的标点

    person = [(129, 105), (108, 154), (147, 156)] + [(80, 121), (93, 110), (93, 125), (109, 121), (147, 121), (161, 111), (160, 125),
                                                     (175, 123)] + [(105, 185), (112, 185), (119, 186), (127, 187), (133, 186), (139, 185), (144, 185)]
    # [(105, 185), (112, 180), (119, 177), (127, 178), (133, 178), (139, 179), (144, 185)] 上嘴唇
    # [(105, 185), (112, 185), (119, 186), (127, 187), (133, 186), (139, 185), (144, 185)] 嘴唇中间
    # [(105, 185), (112, 188), (119, 192), (127, 193), (133, 192), (139, 190), (146, 185)] 下嘴唇

    animal = [(125, 28), (77, 185), (167, 189)] + [(61, 22), (89, 15),  (82, 43), (117, 40), (140, 37), (163, 14), (170, 43),
                                                   (194, 28)] + [(48, 179), (58, 198), (79, 220), (123, 241), (155, 231), (179, 221), (198, 203)]
    # [(48, 179), (58, 198), (79, 220), (123, 241), (155, 231), (179, 221), (198, 203)]

    return animal, person


if __name__ == '__main__':

    animal, person = point_extract()
    im = array(Image.open('./photo.jpg'))
    imshow(im)
    x = [x for x, y in person]
    y = [y for x, y in person]
    plot(x, y, 'r*')
    title('Points of person')
    show()

    im = array(Image.open('./ape.png'))
    imshow(im)
    x = [x for x, y in animal]
    y = [y for x, y in animal]
    plot(x, y, 'r*')
    title('Points of ape')
    show()
