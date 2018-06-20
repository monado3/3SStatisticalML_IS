import numpy as np
import matplotlib.pyplot as plt

plt.subplots_adjust(wspace=0.3,hspace=0.8)

def gauss_2d(x, y):
    x_c = np.array([x, y]) - mu
    return np.exp(- x_c.dot(inv_sigma).dot(x_c[np.newaxis, :].T) / 2.0) / (2*np.pi*np.sqrt(det))

mu = np.array([0,0])
x = y = np.linspace(-7,7,int(1e2))
X, Y = np.meshgrid(x,y)

sigma = np.array([[2,1],
                  [1,4]])
det = np.linalg.det(sigma)
inv_sigma = np.linalg.inv(sigma)
lambdas, phi = np.linalg.eigh(sigma)
Z = np.vectorize(gauss_2d)(X,Y)
plt.subplot(221)
plt.contourf(X,Y,Z)
plt.plot([0,phi[0][0]],[0,phi[0][1]],'k-')
plt.plot([0,phi[1][0]],[0,phi[1][1]],'k-')
plt.axis('equal')
plt.title(f'Σ =\n{sigma}')
print(phi,sep='\n',end='\n\n')

sigma = np.array([[1,-1],
                  [-1,2]])
det = np.linalg.det(sigma)
inv_sigma = np.linalg.inv(sigma)
lambdas, phi = np.linalg.eigh(sigma)
Z = np.vectorize(gauss_2d)(X,Y)
plt.subplot(222)
plt.contourf(X,Y,Z)
plt.plot([0,phi[0][0]],[0,phi[0][1]],'k-')
plt.plot([0,phi[1][0]],[0,phi[1][1]],'k-')
plt.axis('equal')
plt.title(f'Σ =\n{sigma}')
print(phi,sep='\n',end='\n\n')

sigma = np.array([[1,1],
                  [1,10]])
det = np.linalg.det(sigma)
inv_sigma = np.linalg.inv(sigma)
lambdas, phi = np.linalg.eigh(sigma)
Z = np.vectorize(gauss_2d)(X,Y)
plt.subplot(223)
plt.contourf(X,Y,Z)
plt.plot([0,phi[0][0]],[0,phi[0][1]],'k-')
plt.plot([0,phi[1][0]],[0,phi[1][1]],'k-')
plt.axis('equal')
plt.title(f'Σ =\n{sigma}')
print(phi,sep='\n',end='\n\n')

sigma = np.array([[10,-0.5],
                  [-0.5,1]])
det = np.linalg.det(sigma)
inv_sigma = np.linalg.inv(sigma)
lambdas, phi = np.linalg.eigh(sigma)
Z = np.vectorize(gauss_2d)(X,Y)
plt.subplot(224)
plt.contourf(X,Y,Z)
plt.plot([0,phi[0][0]],[0,phi[0][1]],'k-')
plt.plot([0,phi[1][0]],[0,phi[1][1]],'k-')
plt.axis('equal')
plt.title(f'Σ =\n{sigma}')
print(phi,sep='\n',end='\n\n')
plt.savefig('2_2dgauss_by4covmat.png',bbox_inches='tight')
