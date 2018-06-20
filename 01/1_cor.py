import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 1000
x = np.random.randn(n)
y1 =  x +   np.random.randn(n)
y2 = -x +   np.random.randn(n)
y3 =        np.random.randn(n)
y4 = x**2 + np.random.randn(n)

plt.subplots_adjust(wspace=0.3,hspace=0.4)

plt.subplot(221)
plt.grid()
plt.scatter(x,y1,marker='x',s=1)
plt.title('data 1')

plt.subplot(222)
plt.grid()
plt.scatter(x,y2,marker='x',s=1)
plt.title('data 2')

plt.subplot(223)
plt.grid()
plt.scatter(x,y3,marker='x',s=1)
plt.title('data 3')

plt.subplot(224)
plt.grid()
plt.scatter(x,y4,marker='x',s=1)
plt.title('data 4')

plt.savefig('1_data1to4.png')

print(np.corrcoef(x,y1)[0][1])
print(np.corrcoef(x,y2)[0][1])
print(np.corrcoef(x,y3)[0][1])
print(np.corrcoef(x,y4)[0][1])
