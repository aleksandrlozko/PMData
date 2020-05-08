import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.dataloader import DataLoader
import pickle

from sklearn.feature_selection import RFE

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


class Model:

    """
    Preparing the model
    """

    @staticmethod
    def prepare_model(person, predict):

        """
        Choosing of the best model and save it

        :param person: Number of the person for whom the model wii be trained
        :param predict: Feature for prediction
        """

        #preparing data
        main = DataLoader.prepare_data(path_reporting=f'../data/p{person}/googledocs/reporting.csv', path_wellness=f'../data/p{person}/pmsys/wellness.csv',
                                       path_distance=f'../data/p{person}/fitbit/distance.json', path_calories=f'../data/p{person}/fitbit/calories.json')

        Y = main[predict]
        main.drop([predict], axis=1)
        X = main

        n = round(len(X) * 0.7)

        #feature selection
        model = LogisticRegression(max_iter=3000)
        rfe = RFE(model, 3)
        rfe = rfe.fit(X, Y)
        X_new = rfe.fit_transform(X, Y)

        #separation into train and test samples
        X_train = X_new[:n]
        y_train = list(Y[:n])
        X_test = X_new[n:]
        y_test = list(Y[n:])

        train = pd.DataFrame(X_train)
        train['Y'] = y_train
        train.to_csv(f'../data/datasets/train/{predict}/{person}.csv', index=False)

        test = pd.DataFrame(X_test)
        test['Y'] = y_test
        test.to_csv(f'../data/datasets/test/{predict}/{person}.csv', index=False)

        classifiers = [
            KNeighborsClassifier(3),
            SVC(probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LogisticRegression()
        ]


        #choosing of the classifier with the greatest accuracy
        log_cols = ["Classifier", "Accuracy"]
        log = pd.DataFrame(columns=log_cols)

        acc_dict = {}

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, train_predictions)
            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

        for clf in acc_dict:
            acc_dict[clf] = acc_dict[clf]
            log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
            log = log.append(log_entry)

        #plot of Classifier Accuracy
        plt.xlabel('Accuracy')
        plt.title('Classifier Accuracy')

        sns.set_color_codes("muted")
        sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
        plt.show()

        best_classifier = log[log.Accuracy == max(log.Accuracy)].iloc[-1].Classifier

        classifiers_dict = {
            'KNeighborsClassifier': KNeighborsClassifier(3),
            'SVC': SVC(probability=True),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'GaussianNB': GaussianNB(),
            'LogisticRegression': LogisticRegression()
        }

        #save model
        model = classifiers_dict[best_classifier]
        model.fit(X_train, y_train)

        with open(f'../models/{predict}/{person}.pickle', 'wb')as f:
            pickle.dump(model, f)


    @staticmethod
    def predict(person, predict):

        """
        Load the model and predict answers

        :param person: Number of person
        :return: Dict of results
        """

        result = {}

        loaded_model = pickle.load(open(f'../models/{predict}/{person}.pickle', 'rb'))

        test_X = pd.read_csv(f'../data/datasets/test/{predict}/{person}.csv')
        test_y = test_X.Y

        test_X = test_X.drop(['Y'], axis=1)

        pred = loaded_model.predict(test_X)

        result['Accuracy'] = accuracy_score(test_y, pred)
        result['Confution Matrix'] = confusion_matrix(test_y, pred)
        result['Classification Report'] = classification_report(test_y, pred)

        if predict == 'result':
            ans = {
                0: 'Without changes',
                1: 'Descrease',
                2: 'Increase'
            }

            test_y =  [ans[x] for x in list(test_y)]
            pred = [ans[x] for x in list(pred)]

        result['Correct answers'] = test_y
        result['Predicted answers'] = pred

        return result
