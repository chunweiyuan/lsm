from sklearn import svm, grid_search, datasets

class SVMLinear(object):
    """
    This is essentially a wrapper function that combines the
    training, testing, grid search functionalities of sklearn in to one object.
    Customized to Linear SVM (uses svm.LinearSVC).
    """

    def __init__(self, training_set=None, labels=None, parameters=None):
        """
        Given training set and labels, automates the setting of hyperparameters
        via self.gridsearch.
        If not given, then assumes parameters is given to set the classifier.
        Typically parameters can be obtain by taking obj.get_params() from
        another classifier.
        NOTE: the grid search in the __init__ only sets the hyperparameters,
        and the classifier still needs to be fitted to the training data later.
        """
        self.clf = self.gridsearch(training_set, labels) if parameters is None\
              else svm.LinearSVC(**parameters)


    def fit(self, x, y):
        """
        Trains the classifier.
        x: data.
        y: target.
        """
        self.clf.fit(x, y)


    def score(self, x, y):
        """
        Tests the classifier.
        x: data.
        y: target.
        """
        return self.clf.score(x, y)


    def predict(self, x):
        """Perform classification on samples in x"""
        return self.clf.predict(x)


    def get_params(self):
        """Get the hyperparameters of the classifier"""
        return self.clf.get_params()


    def decision_function(self, x):
        """
        Predict confidence scores for sampels.
        The confidence score for a sample is a signed distance of that sample
        to the hyperplane.
        x: array-like n-by-m.  n = number of sampels, m = number of dimensions.
        returns: n-by-m array.
        """
        return self.clf.decision_function(x)


    @classmethod
    def gridsearch(cls, x, y, scoring='f1_weighted'):
        """
        x, y are the training set and labels for the grid search.
        returns the best estimator, where the best C hyperparameter value is set.
        In essence, this performs the cross-validation to determine the hyperparameters.
        """
        if x is None or y is None: raise Exception('Data or labels are None!')
        if len(x) != len(y): raise Exception('len(data) != len(labels)')
        if len(x) < 3 or len(y) < 3: raise Exception('Mininum 3 data points for each class.')
        svr = svm.LinearSVC() # we only care about linear classifier for this class
        cgrid = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] # hardcoding this grid
        parameters = {'C':cgrid}
        clf = grid_search.GridSearchCV(svr, parameters, scoring=scoring)
        clf.fit(x, y)
        for score in clf.grid_scores_:
            print 'Classifier grid search score: {}'.format(score)  # print this out for fun's sake
        return clf.best_estimator_ # this returns the best linearsvc estimator


if __name__ == '__main__':
   iris = datasets.load_iris()
   tempclf = SVMLinear(iris.data, iris.target)
   svmlin = SVMLinear(parameters = tempclf.get_params())
   svmlin.fit(iris.data, iris.target)
   print svmlin.predict(iris.data)
   print svmlin.score(iris.data, iris.target)
