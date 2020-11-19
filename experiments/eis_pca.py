import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pdb

def main():

    directory = 'data/eis/'
    Ts = ['25']
    cells = ['01', '02', '03', '04', '05', '06', '07', '08']
    states = ['I', 'II', 'III']
    n = 60
    n_components = 50
    data = []
    for cell in cells:
        for T in Ts:
            for state in states:

                file = 'EIS_state_{}_{}C{}.txt'.format(state, T, cell)

                df = pd.read_csv(directory + file, delimiter='\t')

                for i in range(int(df.shape[0]/n)):
                    re_z = df['           Re(Z)/Ohm'].loc[i*n:int((i+1)*n - 1)].to_numpy()
                    im_z = df['  -Im(Z)/Ohm'].loc[i*n:int((i+1)*n - 1)].to_numpy()
                    x = np.concatenate((re_z, im_z), axis=0).reshape(-1)

                    data.append(x)

    data = np.array(data)

    scaler = StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    pdb.set_trace()
                
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    ratio = pca.explained_variance_ratio_
    variances = []
    for j in range(1, n_components):
        variances.append(np.sum(ratio[0:j]))

    variances = np.array(variances)

    pdb.set_trace()

    plt.plot(np.arange(1, n_components), variances)
    plt.show()









if __name__ == '__main__':
    main()