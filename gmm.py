#coding:utf-8
import numpy as np
import sys

K = 6
D = 12

def scale(X):
    """データ行列Xを属性ごとに標準化したデータを返す"""
    # 属性の数（=列の数）
    col = X.shape[1]

    # 属性ごとに平均値と標準偏差を計算
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # 属性ごとデータを標準化
    for i in range(col):
        X[:,i] = (X[:,i] - mu[i]) / sigma[i]

    return X

def gaussian(x, mean, cov):
    temp1 = 1 / ((2 * np.pi) ** (x.size/2.0))
    temp2 = 1 / (np.linalg.det(np.diag(cov)) ** 0.5)
    temp3 = - 0.5 * np.dot(np.dot(x - mean, np.linalg.inv(np.diag(cov))), x - mean)
    return temp1 * temp2 * np.exp(temp3)

if __name__ == "__main__":
    data = np.genfromtxt(sys.argv[1])
    X = data[:, 0:D]
    X = scale(X)
    N = len(X)

    # mu, sigma, pi
    mu = np. array([[ 0.81749364, -0.20255307,  0.38149055, -0.01794494,  0.51666461,
         0.08540108, -0.56657963,  0.87723727,  0.37331541, -0.61889183,
         0.52924224,  0.39852004],
       [ 1.14575145, -1.53795642, -1.57995293, -0.72560391,  0.74712722,
        -0.37805792, -0.76581448,  0.21623829, -1.44954708,  1.33369371,
        -0.15545641, -2.09393416],
       [-1.21932976,  1.39862216,  0.77600216,  1.13194579, -1.30792368,
         0.67566378,  0.96994486, -0.6277027 ,  1.18733033, -0.40242254,
         0.52416299,  0.70664962],
       [ 1.14363717, -0.78524447, -1.40907944, -0.45100207,  1.41843591,
        -1.24284407, -1.30995417,  0.88047981, -0.90236124, -1.37983914,
        -0.2806354 , -0.72183315],
       [-0.45575046, -0.05853232,  0.20737562, -0.86886806, -0.02481526,
        -0.71577859,  0.87259374, -1.22546256, -0.50706214,  1.01891822,
        -1.30500843,  0.60438571],
       [-0.77513352,  0.64616189,  0.83804206,  1.24334082, -0.929044  ,
         1.91086189, -0.27446666,  0.97990516,  1.02506399, -0.45971945,
         1.46600082, -0.05075862]])
    sigma = np. array([[ 0.03098865,  0.03528061,  0.14419755,  0.34143348,  0.20050387,
         0.09762261,  0.173794  ,  0.10794781,  0.07920036,  0.07857036,
         0.02999709,  0.15701589],
       [ 0.00793893,  0.04217663,  0.1984885 ,  0.04355858,  0.02656853,
         0.07858338,  1.07816417,  0.05509454,  0.02065847,  0.06617592,
         0.22785552,  0.29165214],
       [ 0.02108283,  0.09543097,  0.26010889,  0.10684005,  0.33798569,
         0.12621997,  0.18031343,  0.01407723,  0.16205948,  0.43041685,
         0.12802325,  0.09642   ],
       [ 0.09315006,  0.01692993,  0.11619844,  0.23417411,  0.50717095,
         0.0780679 ,  0.0455026 ,  0.59363944,  0.2918497 ,  0.10724874,
         0.89650389,  0.28606179],
       [ 0.11001174,  0.64276506,  0.25410369,  0.55207779,  0.10202757,
         0.20944594,  0.1890022 ,  0.19336009,  0.21435304,  0.09944565,
         0.05364207,  0.13563233],
       [ 0.73128766,  0.05724234,  0.17899451,  0.23760281,  0.05354714,
         0.02931033,  0.17383604,  0.25043962,  0.09279665,  0.07632113,
         0.04088429,  0.14486572]])
    pi = np. array([ 0.1875,  0.125 ,  0.1875,  0.125 ,  0.25  ,  0.125 ])

    for n in range(N):
        p = 0.0
        for k in range(K):
            p += pi[k] * gaussian(X[n], mu[k], sigma[k])

        print p
