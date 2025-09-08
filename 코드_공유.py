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

dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 5],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search_dt.fit(X_train, y_train)

print("최적의 하이퍼파라미터 조합:", grid_search_dt.best_params_)
print("최적 교차검증 정확도:", grid_search_dt.best_score_)

best_model = grid_search_dt.best_estimator_
importances = best_model.feature_importances_

plt.figure(figsize=(10,6))
sns.barplot(
    x=importances, 
    y=wine.feature_names, 
    palette="viridis"
)
plt.title("Feature Importance (Decision Tree)", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.show()