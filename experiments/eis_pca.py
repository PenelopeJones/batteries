import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pdb

def main():

    directory = 'data/eis/'
    Ts = ['25']
    cells = ['01']
    states = ['I', 'II', 'III']
    n = 60
    data = []
    for cell in cells:
        for T in Ts:
            for state in states:

                file = 'EIS_state_{}_{}C{}'.format(state, T, cell)

                df = pd.read_csv(directory + file, delimiter='\t')
                pdb.set_trace()

                for i in range(10):
                    pdb.set_trace()
                    re_z = df['Re(Z)/Ohm'].loc[i*n:(i+1)*n].numpy()
                    im_z = df['-Im(Z)/Ohm'].loc[i*n:(i+1)*n].numpy()
                    pdb.set_trace()
                    x = np.concatenate((re_z, im_z), axis=0).reshape(-1)

                    data.append(x)

    data = np.array(data)
    pdb.set_trace()

    n_components = 50
                
    pca = PCA(n_components=n_components)
    pca.fit(data)
    ratio = pca.explained_variance_ratio_
    variances = []
    for j in range(1, n_components):
        variances.append(np.sum(ratio[0:j]))

    variances = np.array(variances)

    plt.plot(np.arange(1, n_components), variances)
    plt.show()









if __name__ == '__main__':
    main()