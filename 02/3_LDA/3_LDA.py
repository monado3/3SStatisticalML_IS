# homework3 at 04/25

import numpy as np
import matplotlib.pyplot as plt

# In[] def the helper function
def mat(array):
    return np.matrix(array)

def plot_cm(confmat):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.title('recognized category')
    plt.ylabel('true category')
    plt.tight_layout()
    plt.savefig('3_confusion_matrix.png')

# In[] custom parameter
np.random.seed(0)
n = 600
alpha = 0.1
mu1 = np.array([2,0])
mu2 = np.array([-2,0])
sigma = np.array([[1,0],[0,9]])

# In[] dependent parameter
n1 = sum(np.random.rand(n) < alpha)
n2 = n - n1
n1

# In[] generate data
cate1 = np.array((sigma[0][0]*np.random.randn(n1)+mu1[0],
                      sigma[1][1]*np.random.randn(n1)+mu1[1]))
cate2 = np.array((sigma[0][0]*np.random.randn(n2) + mu2[0],
                      sigma[1][1]*np.random.randn(n2)+mu2[1]))
cate2.shape

# In[] hyper-plane from Fisher's LDA
a = mat(sigma).I*(mat(mu1).T - mat(mu2).T)
b = -0.5*( mat(mu1)*mat(sigma).I*mat(mu1).T - mat(mu2)*mat(sigma).I*mat(mu2).T) + np.log(n1/n2)
print(f'a =\n{a}')
print(f'b =\n{b}')

# conclusion
# from a.T*x + b = 0, -> x1 = -b/a[0]

# In[] calculate the equation of hyper-plane
x1 = -b[0,0]/a[0,0]
x1

# In[] save the graph
plt.plot( [x1, x1], [-30, 30], 'k-', lw=2)
plt.scatter(cate1[0], cate1[1], marker='x', s=20, label='category1')
plt.scatter(cate2[0], cate2[1], marker='x', s=20, label='category2')
plt.title('homework3 - LDA by Python3')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper right')
plt.savefig('LDA_result.png')

# In[] show the result
cate1_T = np.sum(cate1[0] > x1)
cate2_T = np.sum(cate2[0] < x1)
cate1_F = n1 - cate1_T
cate2_F = n2 - cate2_T
cm = np.array([[cate1_T,cate1_F],[cate2_F,cate2_T]])
plot_cm(cm)
print(f'accuracy_score: {cm.trace()}/{cm.sum()} = {cm.trace()/cm.sum():%}')
