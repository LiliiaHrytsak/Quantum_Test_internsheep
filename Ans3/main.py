from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from Preprocessing_data import Preprocessing_data
from Model_learning import ModelLearning

if __name__ == '__main__':

    # Uploading,making preprocessing,splitting data and scaling numerical features.
    obj_pr = Preprocessing_data()
    data = obj_pr.download_csv('C:/Users/38097/Desktop/Quantum internship/internship_train.csv')
    X_train, X_test, y_train, y_test = obj_pr.split_data(data, 0.9, 'target')
    X_train_scaled, X_test_scaled, num_columns = obj_pr.scale_data(data, ['8', 'target'], X_train, X_test)

    obj_l = ModelLearning()
    # Creating dictionary that will contain all validated models with their RMSE values
    temp_dict = dict()
    # Linear regression with L1 regularization
    model1, rmse1, y_pred1 = obj_l.model_validation(Ridge(), X_train_scaled, y_train, X_test_scaled, y_test,
                                           parameters_dict={'solver': ['svd', 'cholesky', 'lsqr', 'sag'],
                                                            'alpha':  range(10, 110, 10),
                                                            'fit_intercept': [True, False]}, cv=3)
    temp_dict[model1] = rmse1

    #Ensemble learning algorithm Random Decision Forests
    model2, rmse2, y_pred2 = obj_l.model_validation(RandomForestRegressor(random_state=0), X_train_scaled, y_train,
                                           X_test_scaled, y_test,
                                           parameters_dict={'n_estimators': [100,150,200,250,300],
                                                            'max_depth': [1, 2, 3, 4]})
    temp_dict[model2] = rmse2

    # Boosting algorithm AdaBoost regression
    model3, rmse3, y_pred3 = obj_l.model_validation(AdaBoostRegressor(), X_train_scaled, y_train,
                                           X_test_scaled, y_test,
                                           parameters_dict={'n_estimators': [50, 100],
                                                            'learning_rate': [0.01, 0.05, 0.1, 0.5]})
    temp_dict[model3] = rmse3

    best_model = obj_l.choose_model(temp_dict)
    resulted_model, preprocessor = obj_l.train_model(best_model, X_train, X_test, y_train, y_test, num_columns)

    obj_l.make_predictions(resulted_model, 'C:/Users/38097/Desktop/Test Tesk_Internship/'
                                           'Quantum internship/internship_hidden_test.csv',preprocessor,num_columns)
