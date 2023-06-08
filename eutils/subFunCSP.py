import numpy as np


def learnCSP(X,Y):
    nbChannels = len(X[0][0])
    nbTrials = len(X)
    classLables = np.unique(Y)
    nbClasses = len(classLables)

    if nbClasses!=2:
        print('ERROR! GitCSP can only be used for two classes!')
        return
    covMatrices = [np.array([]), np.array([])]
    trialCov = np.zeros((nbTrials, nbChannels, nbChannels))
    for trialNum in range(nbTrials):
        E = X[trialNum,:,:].transpose()
        E_ = E.transpose()
        EE = np.dot(E, E_)
        trialCov[trialNum,:,:] = EE/np.trace(EE)
    del E
    del EE

    for c in range(nbClasses):
        classes = list()
        for i in range(nbTrials):
            if Y[i] == classLables[c]:
                classes.append(trialCov[i])
        classes = np.array(classes)
        covMatrices[c] = np.mean(classes,0)
    covMatrices = np.array(covMatrices)
    covTotal = covMatrices[0] + covMatrices[1]

############## whitening transform of total covariance matrix ###########################
    eigenvalues, Ut = np.linalg.eig(covTotal)
    egIndex = np.argsort(-eigenvalues)
    eigenvalues = sortEigs(egIndex, eigenvalues)
    Ut = sortVectorByEigs(egIndex, Ut)

########### transforming covariance matrix of first class using P ########################
    P = np.dot(np.diag(np.sqrt(1.0/eigenvalues)), np.transpose(Ut))
    tmp = np.dot(covMatrices[0], np.transpose(P))
    transformedCov1 = np.dot(P, tmp)
    # filename = 'transformedCov1.xlsx'
    # write_excels(np.around(transformedCov1, decimals=4), filename)

################## EVD of the transformed covariance matrix ##############################
    eigenvalues, U1 = np.linalg.eig(transformedCov1)
    egIndex = np.argsort(-eigenvalues)
    U1 = sortVectorByEigs(egIndex, U1)
    CSPMatrix = np.dot(np.transpose(U1), P)
    return CSPMatrix

def sortEigs(egIndex, eigenvalues):
    tmp = list()
    for i in range(len(eigenvalues)):
        tmp.append(eigenvalues[i])
    tmp = np.array(tmp)
    for i in range(len(eigenvalues)):
        eigenvalues[i] = tmp[egIndex[i]]

    return eigenvalues

def sortVectorByEigs(egIndex, Ut):
    tmp = list()
    for i in egIndex:
        temp = Ut[:, egIndex]
        tmp.append(temp)
    Ut = np.array(tmp[0])

    return Ut


def csp_dot(w_csp, x):
    # CSP处理过程
    tmp_x = np.empty([x.shape[0], x.shape[1], csp_channel_num * 2, 0])
    # TODO: 多频带处理
    # 准备当前csp矩阵
    tmp_w_csp = w_csp
    tmp_w_csp = np.concatenate([tmp_w_csp[:, 0:csp_channel_num], tmp_w_csp[:, -csp_channel_num:]], axis=1)
    csp_x = np.matmul(x[:, :, :], tmp_w_csp)
    tmp_x = np.concatenate([tmp_x, tf.expand_dims(csp_x, axis=-1)], axis=-1)

    return tmp_x


def main():

    w_csp = learnCSP(x_train, y_train)
    # csp 处理数据
    x_train = csp_dot(w_csp, x_train)
    x_test = csp_dot(w_csp, x_test)
    x_val = csp_dot(w_csp, x_val)