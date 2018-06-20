# homework4 at 04/25

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

# In[] def the helper functions
def mat(array):
    return np.matrix(array)

def recognize(row):
    mahala_arr = np.zeros(10)
    for y in range(0,10):
        x_mu = mat( row.drop('y').values - est_mu.loc[y].values)
        mahala_arr[y] = (-0.5*np.dot(x_mu,sigma_inv_arr[y,:,:])*x_mu.T)[0,0]
    return mahala_arr.argmax()

def plot_cm(confmat):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.title('recognized category')
    plt.ylabel('true category')
    plt.tight_layout()
    plt.savefig('4_confusion_matrix.png')


# In[] load data
DATA_DIR='./digit'

files = glob.glob(os.path.join(DATA_DIR,'*.csv'))
df_train_lis=[]
df_test_lis =[]
for file in files:
    tmp_df = pd.read_csv(file, header=None, encoding='utf-8')
    filename = os.path.basename(file)
    if filename[6:10] == 'test':
        tmp_df['y'] = int(filename[10])
        df_test_lis.append(tmp_df)
    else:
        tmp_df['y'] = int(filename[11])
        df_train_lis.append(tmp_df)
df_train = pd.concat(df_train_lis, ignore_index=True)
df_test = pd.concat(df_test_lis, ignore_index=True)

# In[] estimate the mean and covariance matrix of each category of train
train_grouped = df_train.groupby('y')
est_mu = train_grouped.mean()
est_mu
est_cov = train_grouped.cov()
est_cov

sigma_inv_lis = [] # list of each category(0-9)'s sigma.Inverse
for y in range(0,10):
    sigma = np.matrix(est_cov.loc[y].values + 1e-1*np.eye(256))
    sigma_inv_lis.append(sigma.I)
sigma_inv_arr = np.array(sigma_inv_lis)

sigma_det_lis = [] # list of each category(0-9)'s sigma.determination'
for y in range(0,10):
    sigma_det_lis.append(np.linalg.det(est_cov.loc[y].values))
sigma_det_arr = np.array(sigma_det_lis)
sigma_det_arr # Afterall, I don't use this because all elements were 0.0.

# In[] estimate y
df_test['estimated_y'] = df_test.apply(recognize, axis=1)

# In[] show and save the result
cm = confusion_matrix(df_test['y'], df_test['estimated_y'])
plot_cm(cm)
print(f'accuracy_score: {cm.trace()}/{cm.sum()} = {cm.trace()/cm.sum():%}')
