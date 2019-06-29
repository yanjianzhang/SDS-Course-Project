from itertools import islice
file = open("../w.txt","r")
series = []
for line in islice(file,2,None):
    series.append(float(line.strip()))
# series
from scipy import  stats
import statsmodels.api as sm  # 统计相关的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch  # 条件异方差模型相关的库
# am = arch.arch_model(series, mean='zero', dist="mix", p=1, o=1, q=1, p_=0.5, vol = "EGARCH")
am = arch.arch_model(series, mean='zero', p=1, o=1, q=1, p_=0.5, vol = "EGARCH")
print(am.fit())