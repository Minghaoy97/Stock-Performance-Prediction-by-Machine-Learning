import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as RF
import graphviz

def RF_regression(filename,featuresnames,classnames,t0,t1,k=1,d=0.01,n=5,output=False):
    df = pd.read_csv(filename)

    # Seperate Date
    df['Y'], df['M'], df['Da'] = df["Date"].str.split('-').str
    df['Y'].astype("int")
    df['M'].astype("int")
    df['Da'].astype("int")
    del df['Date']

    # generate label
    def gen_label(data, price='Close', k=1, d=0.01, n=5):
        p0 = np.array(data[price])
        p1 = np.delete(np.append(p0, 0), 0)
        l = p1 / p0
        label = np.zeros(l.shape[0])
        for i in range(0, n):
            label0 = (l > (k + i * d)) + 0
            label = label + label0
        data["label"] = label
        return data

    df = gen_label(data=df,k=k,d=d,n=n)

    #drop N/A
    df.dropna(inplace=True)

    X = np.array(df[featuresnames,'Y','M','Da'])
    Y = np.array(df["label"]).reshape(-1, 1)

    # Seperate train set
    X_train = X[t0:t1, :]
    X_test = X[t1:X.shape[0], :]
    Y_train = Y[t0:t1, :]
    Y_test = Y[t1:Y.shape[0], :]

    # Random forest
    RF0 = RF(max_depth=15, min_samples_leaf=20)
    RF0.fit(X_train, Y_train)
    accuracy0 = RF0.score(X_test, Y_test)
    print(accuracy0)
    Estimators = RF0.estimators_
    if output == True:
        for index, model in enumerate(Estimators):
            filename = 'RF' + str(index)
            dot_data = skl.tree.export_graphviz(model, out_file=None,
                                                filled=True, rounded=True,
                                                special_characters=True,
                                                feature_names=featurenames,
                                                class_names=classnames)
            graph = graphviz.Source(dot_data)
            graph.render(filename=filename)