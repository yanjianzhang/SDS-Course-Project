setwd("D:/SP2(资料存储室)/复旦学习资料/大三下/金融与经济/LAB-2")
library(openxlsx)
library(rugarch)
# data(sp500ret)
# View(sp500ret)
# 文件名+sheet的序号
data<- read.xlsx("w.xlsx", sheet = 1)
data(sp500ret)
ctrl = list(TOL = 1e-8)
ugarchfit(spec = ugarchspec(mean.model = list(armaOrder = c(0, 0),
                                              include.mean = TRUE), variance.model = list(model = "eGARCH", garchOrder = c(1, 1))), 
          solver.control = list(tol = 1e-6),data = data)
egarch.fit = ugarchfit(data = data$w, spec = spec,
                       solver = "solnp", solver.control = ctrl)
fcat = ugarchforecast(egarch.fit, n.ahead=1)
# print(egarch.fit)
# spec = ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
#                   mean.model = list(armaOrder = c(1,1), include.mean = TRUE),
#                   distribution.model = "std", fixed.pars = as.list(coef(egarch.fit)))
# egarch.filter = ugarchfilter(data = data[,1,drop=FALSE], spec = spec)
print(egarch.filter)
GMM.test<- function(data,alpha=0.05,pic=TRUE){
    sol<- shapiro.test(data)
    print(sol)
    if(sol$p.value>alpha){
        print(paste("success:服从该混合正态分布,p.value=",sol$p.value,">",alpha))    
    }else{
        print(paste("fail:不服从该混合分布,p.value=",sol$p.value,"<=",alpha))
    }
    sol
}
res <- GMM.test(data$w)

