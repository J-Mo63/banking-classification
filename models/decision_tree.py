# Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
from utils import pre_processing as prep, classification_helper as clf_helper
import pandas as pd


def build_dt(df_train, df_target):
    # Split the data into training and testing sets of features and targets
    train_features, test_features, train_targets, test_targets = train_test_split(
        df_train, df_target, test_size=0.3, random_state=1)

    # Create a decision tree classifier that sorts??? by entropy and pre-prunes to a max depth
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=5)

    # Train the decision tree based on features
    dt = dt.fit(train_features, train_targets)

    # Get predictions for testing features
    predictions = dt.predict(test_features)

    # Determine accuracy for testing targets and notify console
    clf_helper.display_accuracy(test_targets, predictions, name='Decision Tree')

    # Generate an ROC curve from the predictions
    clf_helper.generate_roc(dt, test_features, test_targets, name='Decision Tree')

    # Visualise the decision tree and export to png
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=list(train_features),
                    class_names=['no', 'yes'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('decision_tree.png')
    Image(graph.create_png())

    # Return the generated decision tree
    return dt


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

    # Create a combined data frame of pre-processed data for analysis
    processed_df = pd.DataFrame({
        'age': df['age'],
        'duration': df['duration'],
        'campaign': df['campaign'],
        'pdays': df['pdays'],  # Removing pdays decreases accuracy in real world
        # 'previous': df['previous'],     removing days bumps by 0.061%
        'emp.var.rate': df['emp.var.rate'],
        'cons.price.idx': df['cons.price.idx'],  # Removing cons.price.idx decreases accuracy
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
        # 'Telephone': binarised_contact,     removing telephone makes no change
        'Default-Yes': binarised_default['yes'],
        'Default-No': binarised_default['no'],
        'Default-Unknown': binarised_default['unknown'],
        # 'Housing-Yes': binarised_housing['yes'],   removing housing makes no change
        # 'Housing-No': binarised_housing['no'],
        # 'Housing-Unknown': binarised_housing['unknown'],
        'Loan-Yes': binarised_loan['yes'],
        'Loan-No': binarised_loan['no'],
        'Loan-Unknown': binarised_loan['unknown'],
        'Previous Success': binarised_poutcome['success'],
        'Previous Failure': binarised_poutcome['failure'],
        'No Previous Contact': binarised_poutcome['nonexistent'],
        # 'Mon': binarised_day['monday'],
        # 'Tue': binarised_day['tuesday'],
        # 'Wed': binarised_day['wednesday'],  removing days bumps by 0.01%
        # 'Thu': binarised_day['thursday'],
        # 'Fri': binarised_day['friday'],
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
        # 'basic.4y': binarised_education['basic.4y'],
        # 'basic.6y': binarised_education['basic.6y'],
        # 'basic.9y': binarised_education['basic.9y'], removing education bumps by 0.024%
        # 'high.school': binarised_education['high.school'],
        # 'illiterate': binarised_education['illiterate'],
        # 'professional.course': binarised_education['professional.course'],
        # 'university.degree': binarised_education['university.degree'],
        # 'unknown': binarised_education['unknown'],
    })

    if include_target:
        processed_df[target_col] = df[target_col]

    return processed_df
