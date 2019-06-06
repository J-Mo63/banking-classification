import sys
import numpy as np
import pandas as pd
from utils import pre_processing as prep
import tensorflow as tf
from pandas import get_dummies
from sklearn.model_selection import train_test_split


def build_nn_manual(df_train, df_target):
    data = pd.concat([df_train, df_target], axis=1)
    cols = data.columns
    features = cols[0:18]
    labels = cols[18]
    print(features)
    print(labels)

    # Well conditioned data will have zero mean and equal variance
    # We get this automatically when we calculate the Z Scores for the data
    data_norm = pd.DataFrame(data)

    for feature in features:
        data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()

    # Show that should now have zero mean
    print("Averages")
    print(data.mean())

    print("\n Deviations")
    # Show that we have equal variance
    print(pow(data.std(), 2))

    # Shuffle The data
    indices = data_norm.index.tolist()
    indices = np.array(indices)
    np.random.shuffle(indices)
    X = data_norm.reindex(indices)[features]
    y = data_norm.reindex(indices)[labels]

    # One Hot Encode as a data frame
    y = get_dummies(y)

    # Generate Training and Validation Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    # Convert to np arrays so that we can use with TensorFlow
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    # Check to make sure split still has 4 features and 3 labels
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    training_size = X_train.shape[1]
    test_size = X_test.shape[1]
    num_features = 18
    num_labels = 2

    num_hidden = 10

    graph = tf.Graph()
    with graph.as_default():
        tf_train_set = tf.constant(X_train)
        tf_train_labels = tf.constant(y_train)
        tf_valid_set = tf.constant(X_test)

        print(tf_train_set)
        print(tf_train_labels)

        # Note, since there is only 1 layer there are actually no hidden layers... but if there were
        # there would be num_hidden
        weights_1 = tf.Variable(tf.truncated_normal([num_features, num_hidden]))
        weights_2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
        # tf.zeros Automaticaly adjusts rows to input data batch size
        bias_1 = tf.Variable(tf.zeros([num_hidden]))
        bias_2 = tf.Variable(tf.zeros([num_labels]))

        logits_1 = tf.matmul(tf_train_set, weights_1) + bias_1
        rel_1 = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(rel_1, weights_2) + bias_2

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_2, labels=tf_train_labels))
        optimizer = tf.train.GradientDescentOptimizer(.005).minimize(loss)

        # Training prediction
        predict_train = tf.nn.softmax(logits_2)

        # Validation prediction
        logits_1_val = tf.matmul(tf_valid_set, weights_1) + bias_1
        rel_1_val = tf.nn.relu(logits_1_val)
        logits_2_val = tf.matmul(rel_1_val, weights_2) + bias_2
        predict_valid = tf.nn.softmax(logits_2_val)

        num_steps = 10001
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            print(loss.eval())
            for step in range(num_steps):
                _, l, predictions = session.run([optimizer, loss, predict_train])

                sys.stdout.write('\r' + 'Training Model: ' + '{0:.0%}'.format(step/num_steps))

                if step % 2000 == 0:
                    # print(predictions[3:6])
                    print('\nLoss at step %d: %f' % (step, l))
                    print('Training accuracy: %.1f%%' % accuracy(predictions, y_train[:, :]))
                    print('Validation accuracy: %.1f%%\n' % accuracy(predict_valid.eval(), y_test))


def build_nn(df_train, df_target):
    train_features, test_features, train_targets, test_targets = train_test_split(
        df_train, df_target, test_size=0.3, random_state=1)

    # Normalise all feature values
    train_features = z_score(train_features)
    test_features = z_score(test_features)

    # Describe the network inputs
    input_columns = []
    for key in train_features.keys():
        input_columns.append(tf.feature_column.numeric_column(key=key))

    # Build a deep neural network to predict classes over hidden layers
    nn = tf.estimator.DNNClassifier(
        feature_columns=input_columns,
        n_classes=2, hidden_units=[10, 10, 10])  # Three hidden layers of 10 nodes each

    # Set a standard batch size for processing
    batch_size = 32

    # Train the model over a number of iterations
    nn.train(
        input_fn=lambda: train_input_fn(train_features, train_targets, batch_size),
        steps=10000)  # 10k iterations for training

    # Evaluate the model on the test data
    eval_result = nn.evaluate(
        input_fn=lambda: pred_input_fn(test_features, test_targets, batch_size))

    # Display the accuracy on the test set
    print('Neural Network Accuracy: {accuracy:0.3%}'.format(**eval_result))

    return nn


def predict(nn, data):
    # Set a standard batch size for processing
    batch_size = 32

    # Normalise all feature values
    data = z_score(data)

    # Get predictions for the provided data
    pred_results = nn.predict(
        input_fn=lambda: pred_input_fn(data, None, batch_size))

    # Extract predictions from the generator object
    predictions = []
    for i in range(len(data)):
        predictions.append(next(pred_results)['class_ids'][0])

    # Return a numpy array of predictions
    return np.array(predictions)


def train_input_fn(features, targets, batch_size):
    # Format the io dataset from features and targets
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), targets))

    # Shuffle, repeat, and batch the data
    return dataset.shuffle(1000).repeat().batch(batch_size)


def pred_input_fn(features, targets, batch_size):
    # Format the data based on provided attributes
    features = dict(features)
    if targets is None:
        inputs = features
    else:
        inputs = (features, targets)

    # Format the io dataset from features and targets
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch and return the data
    return dataset.batch(batch_size)


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
        'pdays': df['pdays'],
        'previous': df['previous'],
        'emp.var.rate': df['emp.var.rate'],
        'cons.price.idx': df['cons.price.idx'],
        'cons.conf.idx': df['cons.conf.idx'],
        'euribor3m': df['euribor3m'],
        'nr.employed': df['nr.employed'],
        # 'Married': binarised_marital['married'],
        # 'Single': binarised_marital['single'],
        # 'Divorced': binarised_marital['divorced'],
        # 'Admin': binarised_job['admin'],
        # 'Blue-Collar': binarised_job['blue-collar'],
        # 'Entrepreneur': binarised_job['entrepreneur'],
        # 'Housemaid': binarised_job['housemaid'],
        # 'Management': binarised_job['management'],
        # 'Retired': binarised_job['retired'],
        # 'Self-Employed': binarised_job['self-employed'],
        # 'Services': binarised_job['services'],
        # 'Student': binarised_job['student'],
        # 'Technician': binarised_job['technician'],
        # 'Unemployed': binarised_job['unemployed'],
        # 'Telephone': binarised_contact,
        # 'Default-Yes': binarised_default['yes'],
        # 'Default-No': binarised_default['no'],
        # 'Default-Unknown': binarised_default['unknown'],
        # 'Housing-Yes': binarised_housing['yes'],
        # 'Housing-No': binarised_housing['no'],
        # 'Housing-Unknown': binarised_housing['unknown'],
        # 'Loan-Yes': binarised_loan['yes'],
        # 'Loan-No': binarised_loan['no'],
        # 'Loan-Unknown': binarised_loan['unknown'],
        # 'Previous-Success': binarised_poutcome['success'],
        # 'Previous-Failure': binarised_poutcome['failure'],
        # 'No-Previous-Contact': binarised_poutcome['nonexistent'],
        # 'Mon': binarised_day['monday'],
        # 'Tue': binarised_day['tuesday'],
        # 'Wed': binarised_day['wednesday'],
        # 'Thu': binarised_day['thursday'],
        # 'Fri': binarised_day['friday'],
        # 'Mar': binarised_month['march'],
        # 'Apr': binarised_month['april'],
        # 'May': binarised_month['may'],
        # 'Jun': binarised_month['june'],
        # 'Jul': binarised_month['july'],
        # 'Aug': binarised_month['august'],
        # 'Sep': binarised_month['september'],
        # 'Oct': binarised_month['october'],
        # 'Nov': binarised_month['november'],
        # 'Dec': binarised_month['december'],
        # 'basic.4y': binarised_education['basic.4y'],
        # 'basic.6y': binarised_education['basic.6y'],
        # 'basic.9y': binarised_education['basic.9y'],
        # 'high.school': binarised_education['high.school'],
        # 'illiterate': binarised_education['illiterate'],
        # 'professional.course': binarised_education['professional.course'],
        # 'university.degree': binarised_education['university.degree'],
        # 'unknown': binarised_education['unknown'],
    })

    if include_target:
        processed_df[target_col] = df[target_col]

    return processed_df


def accuracy(predictions, labels):
    # Return the computed accuracy for a set of predictions
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def z_score(data):
    # Disable pandas chained assignment warnings
    pd.options.mode.chained_assignment = None

    # Divide the mean-negated value by the set's standard deviation
    for feature in data:
        data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()
    return data
