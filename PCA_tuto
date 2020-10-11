from sklearn.datasets import load_breast_cancer
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime
from datetime import date
from plotly import __version__
%matplotlib inline
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
cf.go_offline()

#data import
breast = load_breast_cancer() 
breast_data = breast.data

#Check the dataset shape
breast_data.shape

#Labels import
breast_labels = breast.target
breast_labels.shape

#Labels addition to our dataset
labels = np.reshape(breast_labels, (569,1))
final_breast_data = np.concatenate([breast_data, labels], axis = 1)
final_breast_data.shape

#Reformat of our dataset with features as column names
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features
features_labels = np.append(features, 'label')
breast_dataset.columns = features_labels
breast_dataset.head()

#Features normalization
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) 

#Standardization results verification
x.shape
np.mean(x), np.std(x)

#Data normalization
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x, columns = feat_cols)
normalised_breast.tail()

#PCA application
pca_breast = PCA(n_components = 2)
principalComponents_breast = pca_breast.fit_transform(x)

principal_breast_Df = pd.DataFrame(data = principalComponents_breast, columns = ['principal component 1', 'principal component 2'])

principal_breast_Df.tail()

#Results
print('Explained variation per principal component:  {}'.format(pca_breast.explained_variance_ratio_))

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

plt.scatter(principal_breast_Df['principal component 1'], principal_breast_Df['principal component 2'], 
            s=10, c=principal_breast_Df['colors'], cmap='viridis')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2 ')
