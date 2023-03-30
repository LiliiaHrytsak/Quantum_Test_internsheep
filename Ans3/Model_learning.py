import pandas as pd
import pickle


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class ModelLearning:

    def choose_model(self,dict_models):
        best_model = min(dict_models, key=dict_models.get)
        rmse = min([min(dict_models.values()) for dict in dict_models])
        print('The best model is', best_model, 'RMSE =', rmse)
        return best_model

    def model_validation(self, certain_model, X_train, y_train, X_test, y_test, parameters_dict={}, cv=4):
        model = certain_model
        clf = GridSearchCV(model, parameters_dict, cv=cv, scoring='neg_root_mean_squared_error')
        validated_model = clf.fit(X_train, y_train)
        y_pred = validated_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f'Best parameters of {certain_model}:{clf.best_params_} \n'
            f'RMSE of {certain_model}: {rmse}')
        return validated_model.best_estimator_, rmse, y_pred

    def train_model(self,best_estimator, X_train, X_test, y_train, y_test,num_columns):
            full_X = pd.concat([X_train, X_test], axis=0)
            full_Y = pd.concat([y_train, y_test], axis=0)
            Scaler = StandardScaler()
            Scaler.fit_transform(full_X[num_columns])
            resulted_model = best_estimator.fit(full_X, full_Y)
            finale_model = 'finalized_model.sav'
            pickle.dump(resulted_model, open(finale_model, 'wb'))
            return finale_model, Scaler

    def make_predictions(self, finale_model, X_test_path, Scaler, num_columns):
        new_data = pd.read_csv(X_test_path)
        Scaler.transform(new_data[num_columns])
        loaded_model = pickle.load(open(finale_model, 'rb'))
        predictions = loaded_model.predict(new_data)
        res = pd.DataFrame(predictions)
        res.index = new_data.index  # its important for comparison
        res.columns = ["prediction"]
        res.to_csv("prediction_results.csv")
