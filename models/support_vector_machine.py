from sklearn.model_selection import train_test_split
from sklearn import svm as support_vector_machine
import pandas as pd
from utils import pre_processing as prep, classification_helper as clf_helper


def build_svm(df_train, df_target):
    # Split the data into training and testing sets of features and targets
    train_features, test_features, train_targets, test_targets = train_test_split(
        df_train, df_target, test_size=0.3, random_state=1)

    # Create an SVM Classifier
    svm = support_vector_machine.SVC(kernel='linear', C=0.5, probability=True)

    # Train the model using the training sets
    svm.fit(train_features, train_targets)

    # Get predictions for testing features
    predictions = svm.predict(test_features)

    # Determine accuracy for testing targets and notify console
    clf_helper.display_accuracy(test_targets, predictions, name='Support Vector Machine')

    # Generate an ROC curve from the predictions
    clf_helper.generate_roc(svm, test_features, test_targets, name='Support Vector Machine')

    # Return the generated decision tree
    return svm


def process_data(df, target_col, include_target=False):
    # Perform binarisation for 'contact'
    binarised_contact = prep.binarise_contact(df['contact'])

    # Perform binarisation for 'loan'
    binarised_loan = prep.binarise_y_n_u(df['loan'])

    # Perform binarisation for 'poutcome'
    binarised_poutcome = prep.binarise_poutcome(df['poutcome'])

    # Perform binarisation for 'month'
    binarised_month = prep.binarise_month(df['month'])

    # Create a combined data frame of pre-processed data for analysis
    processed_df = pd.DataFrame({
        'age': df['age'],
        'duration': df['duration'],
        'campaign': df['campaign'],
        'pdays': df['pdays'],
        'emp.var.rate': df['emp.var.rate'],
        'cons.price.idx': df['cons.price.idx'],
        'cons.conf.idx': df['cons.conf.idx'],
        'euribor3m': df['euribor3m'],
        'nr.employed': df['nr.employed'],
        'Telephone': binarised_contact,
        'Loan-Yes': binarised_loan['yes'],
        'Loan-No': binarised_loan['no'],
        'Loan-Unknown': binarised_loan['unknown'],
        'Previous Success': binarised_poutcome['success'],
        'Previous Failure': binarised_poutcome['failure'],
        'No Previous Contact': binarised_poutcome['nonexistent'],
        'Mar': binarised_month['march'],
        'Apr': binarised_month['april'],
        'May': binarised_month['may'],
        'Jun': binarised_month['june'],
        'Jul': binarised_month['july'],
        'Aug': binarised_month['august'],
        'Sep': binarised_month['september'],
        'Oct': binarised_month['october'],
        'Nov': binarised_month['november'],
        'Dec': binarised_month['december'],
    })

    if include_target:
        processed_df[target_col] = df[target_col]

    return processed_df
