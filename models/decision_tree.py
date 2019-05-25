# Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def build_tree(df_train, df_target):
    # Split the data into training and testing sets of features and targets
    train_features, test_features, train_targets, test_targets = train_test_split(
        df_train, df_target, test_size=0.3, random_state=1)

    # Train the decision tree based on features
    dt = DecisionTreeClassifier().fit(train_features, train_targets)

    # Get predictions for testing features
    predictions = dt.predict(test_features)

    # Determine accuracy for testing targets and notify console
    accuracy = metrics.accuracy_score(test_targets, predictions)
    print("Decision Tree Accuracy: " + "{0:.3%}".format(accuracy))

    # Return the generated decision tree
    return dt
