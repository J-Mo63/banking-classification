from sklearn import metrics
import scikitplot as skplt
import matplotlib.pyplot as plt


def display_accuracy(test_targets, predictions, name=''):
    accuracy = metrics.accuracy_score(test_targets, predictions)
    print(name + 'Accuracy: ' + '{0:.3%}'.format(accuracy))


def generate_roc(clf, test_features, test_targets, name=''):
    predicted_probabilities = clf.predict_proba(test_features)
    skplt.metrics.plot_roc(test_targets, predicted_probabilities,
                           title=(name + 'ROC by Class'), cmap='tab10',
                           plot_micro=False, plot_macro=False)
    plt.show()
