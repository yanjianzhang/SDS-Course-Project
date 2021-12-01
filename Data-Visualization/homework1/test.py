import matplotlib.pyplot as plt


def linear(value):
    r1, s1 = (0.4, 0.2)
    r2, s2 = (0.5, 0.8)
    # print(value)
    if value <= r1:
        return float(s1)/r1*value
    elif value <= r2:
        return s1 + float(s2-s1)/(r2-r1)*(value-r1)
    else:
        return s2 + float(1.0-s2)/(1.0-r2)*(value-r2)


print([linear(i / 10) for i in range(0, 10, 1)])
plt.plot(range(0, 10, 1), [linear(i / 10) for i in range(0, 10, 1)])
plt.show()
