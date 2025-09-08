import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt


wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target']=wine.target
X = df.drop(columns=['target'])
y = df['target']

# train / test 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)