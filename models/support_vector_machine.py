from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm as support_vector_machine
import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
from utils import pre_processing as prep


def build_svm(df_train, df_target):
    # Split the data into training and testing sets of features and targets
    train_features, test_features, train_targets, test_targets = train_test_split(
        df_train, df_target, test_size=0.3, random_state=1)

    # Create an SVM Classifier
    svm = support_vector_machine.SVC(kernel='poly', probability=True)

    # Train the model using the training sets
    svm.fit(train_features, train_targets)

    # Get predictions for testing features
    predictions = svm.predict(test_features)

    # Determine accuracy for testing targets and notify console
    accuracy = metrics.accuracy_score(test_targets, predictions)
    print('Support Vector Machine Accuracy: ' + '{0:.3%}'.format(accuracy))

    # Generate an ROC curve from the predictions
    predicted_probabilities = svm.predict_proba(test_features)
    skplt.metrics.plot_roc(test_targets, predicted_probabilities,
                           title='Decision Tree ROC by Class', cmap='tab10',
                           plot_micro=False, plot_macro=False)
    plt.show()

    # Return the generated decision tree
    return svm


def process_data(df, target_col, include_target=False):
    # Perform binarisation for 'marital'
    binarised_marital = prep.binarise_marital(df['marital'])

    # Perform binarisation for 'contact'
    binarised_contact = prep.binarise_contact(df['contact'])

    # Perform binarisation for 'job'
    binarised_job = prep.binarise_job(df['job'])

    # Perform binarisation for 'education'
    binarised_education = prep.binarise_education(df['education'])

    # Perform binarisation for 'default'
    binarised_default = prep.binarise_y_n_u(df['default'])

    # Perform binarisation for 'housing'
    binarised_housing = prep.binarise_y_n_u(df['housing'])

    # Perform binarisation for 'loan'
    binarised_loan = prep.binarise_y_n_u(df['loan'])

    # Perform binarisation for 'poutcome'
    binarised_poutcome = prep.binarise_poutcome(df['poutcome'])

    # Perform binarisation for 'month'
    binarised_month = prep.binarise_month(df['month'])

    # Perform binarisation for 'day'
    binarised_day = prep.binarise_day(df['day_of_week'])

    # Create a combined data frame of pre-processed data for analysis
    processed_df = pd.DataFrame({
        'age': df['age'],
        'duration': df['duration'],
        'campaign': df['campaign'],
        'pdays': df['pdays'],
        'previous': df['previous'],
        'emp.var.rate': df['emp.var.rate'],
        'cons.price.idx': df['cons.price.idx'],
        'cons.conf.idx': df['cons.conf.idx'],
        'euribor3m': df['euribor3m'],
        'nr.employed': df['nr.employed'],
        'Married': binarised_marital['married'],
        'Single': binarised_marital['single'],
        'Divorced': binarised_marital['divorced'],
        'Admin': binarised_job['admin'],
        'Blue-Collar': binarised_job['blue-collar'],
        'Entrepreneur': binarised_job['entrepreneur'],
        'Housemaid': binarised_job['housemaid'],
        'Management': binarised_job['management'],
        'Retired': binarised_job['retired'],
        'Self-Employed': binarised_job['self-employed'],
        'Services': binarised_job['services'],
        'Student': binarised_job['student'],
        'Technician': binarised_job['technician'],
        'Unemployed': binarised_job['unemployed'],
        'Telephone': binarised_contact,
        'Default-Yes': binarised_default['yes'],
        'Default-No': binarised_default['no'],
        'Default-Unknown': binarised_default['unknown'],
        'Housing-Yes': binarised_housing['yes'],
        'Housing-No': binarised_housing['no'],
        'Housing-Unknown': binarised_housing['unknown'],
        'Loan-Yes': binarised_loan['yes'],
        'Loan-No': binarised_loan['no'],
        'Loan-Unknown': binarised_loan['unknown'],
        'Previous Success': binarised_poutcome['success'],
        'Previous Failure': binarised_poutcome['failure'],
        'No Previous Contact': binarised_poutcome['nonexistent'],
        'Mon': binarised_day['monday'],
        'Tue': binarised_day['tuesday'],
        'Wed': binarised_day['wednesday'],
        'Thu': binarised_day['thursday'],
        'Fri': binarised_day['friday'],
        'Mar': binarised_month['march'],  # Removing months decreases accuracy
        'Apr': binarised_month['april'],
        'May': binarised_month['may'],
        'Jun': binarised_month['june'],
        'Jul': binarised_month['july'],
        'Aug': binarised_month['august'],
        'Sep': binarised_month['september'],
        'Oct': binarised_month['october'],
        'Nov': binarised_month['november'],
        'Dec': binarised_month['december'],
        'basic.4y': binarised_education['basic.4y'],
        'basic.6y': binarised_education['basic.6y'],
        'basic.9y': binarised_education['basic.9y'],
        'high.school': binarised_education['high.school'],
        'illiterate': binarised_education['illiterate'],
        'professional.course': binarised_education['professional.course'],
        'university.degree': binarised_education['university.degree'],
        'unknown': binarised_education['unknown'],
    })

    if include_target:
        processed_df[target_col] = df[target_col]

    return processed_df
