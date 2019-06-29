data=read.csv("~/Desktop/金大final/2000.csv",sep=';',fileEncoding = 'UTF-8',header = TRUE)
pal <- colorFactor(c("navy", "red","#B9D3EE","#FFFAFA","#FFFF00","#00FF7F","#4682B4","#D2B486",
                     "#FFA54F","#FFE1FF","#7D26CD","#FF82AB","#AEEEEE","#FA8072","#C0FF3E","#FFB6C1",
                     "#EEDD82","#EE00EE","#4D4D4D","#CD661D"), domain = data$modularity_class)
data$total <- rowSums(data[,5:13])
x=aggregate(data[,18],list(data$modularity_class),sum)
c=c(pal('0'),pal('1'),pal('2'),pal('3'),pal('4'),pal('5'),pal('6'),pal('7'),pal('8'),pal('9'),
    pal('10'),pal('11'),pal('12'),pal('13'),pal('14'),pal('15'),pal('16'),pal('17'),pal('18'),pal('19'))
library(ggplot2)
ggplot(x, aes(x = Group.1, y = x, fill = c)) + geom_bar(stat = "identity")