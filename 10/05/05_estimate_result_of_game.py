import os.path as op

import numpy as np


class BayesianTable:
    def __init__(self, table):
        self.table = table
        self.num_M = table.shape[0]
        self.num_N = table.shape[1]
        self.M = np.ones(self.num_M)
        self.N = np.ones(self.num_N)
        self.alpha = 1
        self.beta = 1

    def update_M_and_alpha(self):
        denominator = 1 + np.sum(self.beta ** 2 + np.power(self.N, 2))
        self.alpha = np.sqrt(1 / denominator)
        self.M = np.sum(self.table * self.N, axis=1) / denominator

    def update_N_and_beta(self):
        denominator = 1 + np.sum(self.alpha ** 2 + np.power(self.M, 2))
        self.beta = np.sqrt(1 / denominator)
        self.N = np.sum(self.table * self.M[:, np.newaxis], axis=0) / denominator

    def predict(self, row, col):
        E_x_ij = self.M[row] * self.N[col]
        output_fstr = f'E[ x[{row+1},{col+1}] ] : {E_x_ij:.5f}'
        if E_x_ij > 0:
            output_fstr += ' (Win)'
        else:
            output_fstr += ' (Lose)'
        print(output_fstr)

        # dirpath = op.dirname(op.abspath(__file__))
        # with open(op.join(dirpath, 'result.txt'), 'a') as f:
        #     print(output_fstr, file=f)

    def updates(self):
        for loop in range(100):
            self.update_M_and_alpha()
            self.update_N_and_beta()


def main():
    win, lose, unknown = 1, -1, 0
    bayesian_table = BayesianTable(np.array([[win, win, lose, lose],
                                             [unknown, win, lose, lose],
                                             [lose, lose, win, win],
                                             [unknown, lose, unknown, win]]))
    bayesian_table.updates()
    for row, col in ((1, 0), (3, 0), (3, 2)):
        bayesian_table.predict(row, col)


if __name__ == '__main__':
    main()
