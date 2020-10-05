# everything should be inside cross validation loop
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__=='__main__':
    df = pd.read_csv("archive/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    # classifier = ensemble.RandomForestClassifier(n_jobs=-1) # with -1, all the cores will be used
    # param_grid = {
    #     "n_estimators": [100, 200],
    #     "max_depth": [1, 3],
    #     "criterion": ["gini", "entropy"],
    # }

#     model = model_selection.GridSearchCV(
#         estimator=classifier,
#         param_grid=param_grid,
#         scoring="accuracy",
#         verbose=10,
#         n_jobs=1,
#         cv=5,
#     )
    # instead of grid search, you can use randomized search.
    classifier = ensemble.RandomForestClassifier(n_jobs=-1) # with -1, all the cores will be used
    param_grid = {
        "n_estimators": np.arange(100,1500,100),
        "max_depth": np.arange(1,20),
        "criterion": ["gini", "entropy"],
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5,
    )
    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())