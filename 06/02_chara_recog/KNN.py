import glob
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


# In[]
def load_to_df(data_dir='digit'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(current_dir, data_dir, '*.csv'))
    df_train_lis = []
    df_test_lis = []
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
    return df_train, df_test


def grid_search(train_df, classifier, k_set=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
    result_dict = lcv(train_df, classifier, sorted(k_set))
    print(result_dict)
    return max(result_dict.items(), key=lambda x: x[1])[0]


def lcv(df, classifier, k_set, n_split=4):
    num_classes_y = 10
    accuracy_dict = {k: [] for k in k_set}
    grouped = df.groupby('y')
    grouped_df = grouped.apply(lambda x: x.sample(frac=1, random_state=0))
    df = grouped_df.reset_index(drop=True)
    split_df_arr = np.empty((num_classes_y, n_split), dtype=object)
    for y, df_by_y in df.groupby('y'):
        step = df_by_y.shape[0] // n_split
        split_row_start = 0
        for idx in range(n_split):
            split_df_arr[y, idx] = df_by_y.iloc[split_row_start:split_row_start + step, :]
            split_row_start += step
        assert split_row_start == df_by_y.shape[0]
    split_df_lis = []
    for j in range(n_split):
        split_df_lis.append(pd.concat(split_df_arr[:, j], ignore_index=True))
    for j in range(n_split):
        predict_df = split_df_lis[j]
        split_df_lis_cp = split_df_lis[:]
        split_df_lis_cp.pop(j)
        samples_df = pd.concat(split_df_lis_cp, ignore_index=True)
        result_dict = classifier(samples_df, predict_df, k_set)
        for k, accuracy in result_dict.items():
            accuracy_dict[k].append(accuracy)
    for k, accuracy_lis in accuracy_dict.items():
        accuracy_dict[k] = np.mean(accuracy_lis)
    plot_lcv(accuracy_dict, n_split)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'lcv_by_k.png'))
    plt.show()
    return accuracy_dict


def knn(samples_df, predict_df, k_set, output_result=False):
    def recognize(row):
        distance = samples_df.iloc[:, :256].values - row[:256].values
        distance = np.power(distance, 2)
        distance = np.sum(distance, axis=1)
        label_by_nearer = []
        for idx in np.argsort(distance):
            label_by_nearer.append(samples_df['y'][idx])
        label_by_k = []
        for k in k_set:
            counter = Counter(label_by_nearer[:k])
            label_by_k.append(counter.most_common()[0][0])
        return pd.Series({k: label for k, label in zip(k_set, label_by_k)})

    predict_df = pd.concat([predict_df.iloc[:, -1], predict_df.apply(recognize, axis=1)], axis=1)
    if output_result:
        k = k_set[0]
        cm = confusion_matrix(predict_df['y'], predict_df[k])
        plot_cm(cm, k)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(current_dir, 'confmat_by_best_k.png'))
        plt.show()
        print(f'accuracy_score: {cm.trace()}/{cm.sum()} = {cm.trace()/cm.sum():%} (k = {k})')
    else:
        result_dict = {}
        for k in k_set:
            result_dict[k] = accuracy_score(predict_df['y'], predict_df[k])
        return result_dict


def plot_lcv(dic, n_split):
    x = list(dic.keys())
    y = list(dic.values())
    plt.plot(x, y, marker='.')
    plt.title(f'the accuracy score by k (n_split = {n_split})')
    plt.xticks(np.arange(1, 11))
    plt.xlabel('k')
    plt.ylabel('accuracy score')
    plt.grid()


def plot_cm(confmat, k):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.title(f'recognized category (k = {k})')
    plt.ylabel('true category')
    plt.tight_layout()


# In[]
train_df, test_df = load_to_df()
best_k = grid_search(train_df, knn)
knn(train_df, test_df, [best_k], output_result=True)
