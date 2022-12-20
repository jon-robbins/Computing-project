from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
import numpy as np

class ModelOutput:
    def __init__(self, model_name, x_train, y_train):
        self.model_name = model_name
        if model_name == 'Linear':
            self.model = LinearRegression()
        elif model_name == 'Logistic':
            self.model = LogisticRegression()
        elif model_name == 'DecisionTree':
            self.model = DecisionTreeClassifier(max_depth=4, random_state=1)
        elif model_name == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=9, random_state=0)
        elif model_name == 'XGBoost':
            self.model = xgb.XGBClassifier()
        elif model_name == 'RidgeCV':
            alphas = np.arange(2, 100, 0.4)
            self.model = RidgeCV(alphas, normalize = False)
        elif model_name == 'LassoCV':
            self.model = LassoCV(eps = 0.001, n_alphas = 100, normalize = False)
        else:
            self.model = None()

        self.model.fit(x_train, y_train)

        feature_names = x_train.columns
        model_coefficients = self.model.coef_

        coefficients_df = pd.DataFrame(data = model_coefficients,
                                       index = feature_names,
                                       columns = ['Coefficient value'])
        print(coefficients_df)

    def print_r2_score(self, x_test, y_test):
        r2_score_model = None
        if self.model_name == 'RidgeCV':
            r2_score_model = self.model.score(x_test, y_test)
        elif self.model_name == 'LassoCV':
            r2_score_model = self.model.score(x_test, y_test)
        else:
            r2_score_model = r2_score(y_test, self.model.predict(x_test))
        print("R2 of "+self.model_name+" model on train set: {:.2f}".format(r2_score_model))

    def regression_output(self, x_predict):
        return self.model.predict(x_predict)

