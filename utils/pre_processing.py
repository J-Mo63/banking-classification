import pandas as pd
from sklearn import preprocessing


def process_data(df):
    # Perform binarisation for 'marital'
    binarised_marital = binarise_marital(df['marital'])

    # Perform binarisation for 'contact'
    binarised_contact = binarise_contact(df['contact'])

    # Perform binarisation for 'job'
    binarised_job = binarise_job(df['job'])

    # Perform binarisation for 'education'
    binarised_education = binarise_education(df['education'])

    # Perform binarisation for 'default'
    binarised_default = binarise_y_n_u(df['default'])

    # Perform binarisation for 'housing'
    binarised_housing = binarise_y_n_u(df['housing'])

    # Perform binarisation for 'loan'
    binarised_loan = binarise_y_n_u(df['loan'])

    # Perform binarisation for 'poutcome'
    binarised_poutcome = binarise_poutcome(df['poutcome'])

    # Perform binarisation for 'day_of_week'
    binarised_day = binarise_day(df['day_of_week'])

    # Perform binarisation for 'month'
    binarised_month = binarise_month(df['month'])

    # Create a combined data frame of pre-processed data for analysis
    return pd.DataFrame({
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
        'basic.4y': binarised_education['basic.4y'],
        'basic.6y': binarised_education['basic.6y'],
        'basic.9y': binarised_education['basic.9y'],
        'high.school': binarised_education['high.school'],
        'illiterate': binarised_education['illiterate'],
        'professional.course': binarised_education['professional.course'],
        'university.degree': binarised_education['university.degree'],
        'unknown': binarised_education['unknown'],
    })


def bin_equi_width(df, bins):
    # Cut up the data by width
    binned_equi_width = pd.cut(df, bins)

    # Format and listify equi-width list
    bin_equi_width_list = []
    for i in range(binned_equi_width.size):
        left_item = "{0:.2f}".format(binned_equi_width[i].left)
        right_item = "{0:.2f}".format(binned_equi_width[i].right)
        bin_equi_width_list.append([left_item, right_item])

    # Return the results as a list
    return bin_equi_width_list


def bin_equi_depth(df, bins):
    # Cut up the data by width & depth
    binned_equi_depth = pd.qcut(df, bins, duplicates='drop')

    # Format and listify equi-depth list
    bin_equi_depth_list = []
    for i in range(binned_equi_depth.size):
        left_item = "{0:.2f}".format(binned_equi_depth[i].left)
        right_item = "{0:.2f}".format(binned_equi_depth[i].right)
        bin_equi_depth_list.append([left_item, right_item])

    # Return the results as a list
    return bin_equi_depth_list


def normalise_min_max(df):
    # Make the data frame two dimensional
    df = df.values.reshape(-1, 1)

    # Fit the min-max normalised list
    min_max_scaled = preprocessing.MinMaxScaler().fit_transform(df)

    # Return the results as a list
    return pd.DataFrame(min_max_scaled).values.flatten()


def normalise_z_score(df):
    # Make the data frame two dimensional
    df = df.values.reshape(-1, 1)

    # Fit the min-max normalised list
    z_score_scaled = preprocessing.StandardScaler().fit_transform(df)

    # Return the results as a list
    return pd.DataFrame(z_score_scaled).values.flatten()


def discretise_age(df):
    # Isolate the values from the data frame
    df = df.values

    # Discretise the age into categories
    discretised_list = []
    for i in range(df.size):
        item = df[i]
        if item < 45:
            discretised_list.append('Adult')
        elif item <= 65:
            discretised_list.append('Mid-age')
        elif item > 65:
            discretised_list.append('Old-age')
        else:
            discretised_list.append('NaN')

    # Return the results as a list
    return discretised_list


def binarise_marital(df):
    # Isolate the values from the data frame
    df = df.values

    # Binarise the categories
    binarised = preprocessing.LabelBinarizer().fit_transform(df)

    # Format the binarised items into three columns
    binarised_married_list = []
    binarised_single_list = []
    binarised_divorced_list = []
    for i in range(len(binarised)):
        item = binarised[i]
        binarised_divorced_list.append(item[0])
        binarised_married_list.append(item[1])
        binarised_single_list.append(item[2])

    # Return the results as a dictionary
    return {'married': binarised_married_list,
            'single': binarised_single_list,
            'divorced': binarised_divorced_list}


def binarise_poutcome(df):
    # Isolate the values from the data frame
    df = df.values

    # Binarise the categories
    binarised = preprocessing.LabelBinarizer().fit_transform(df)

    # Format the binarised items into three columns
    binarised_failure_list = []
    binarised_nonexistent_list = []
    binarised_success_list = []
    for i in range(len(binarised)):
        item = binarised[i]
        binarised_failure_list.append(item[0])
        binarised_nonexistent_list.append(item[1])
        binarised_success_list.append(item[2])

    # Return the results as a dictionary
    return {'failure': binarised_failure_list,
            'nonexistent': binarised_nonexistent_list,
            'success': binarised_success_list}


def binarise_contact(df):
    # Isolate the values from the data frame
    df = df.values

    # Binarise the categories
    binarised = preprocessing.LabelBinarizer().fit_transform(df)

    # Format the binarised items into one boolean column
    binarised_tel_cell_list = []
    for i in range(len(binarised)):
        item = binarised[i]
        binarised_tel_cell_list.append(item[0])

    # Return the results as a list
    return binarised_tel_cell_list


def binarise_y_n_u(df):
    # Isolate the values from the data frame
    df = df.values

    # Binarise the categories
    binarised = preprocessing.LabelBinarizer().fit_transform(df)

    # Format the binarised items into three columns
    binarised_no_list = []
    binarised_unknown_list = []
    binarised_yes_list = []
    for i in range(len(binarised)):
        item = binarised[i]
        binarised_no_list.append(item[0])
        binarised_unknown_list.append(item[1])
        binarised_yes_list.append(item[2])

    # Return the results as a dictionary
    return {'yes': binarised_yes_list,
            'no': binarised_no_list,
            'unknown': binarised_unknown_list}


def binarise_education(df):
    # Isolate the values from the data frame
    df = df.values

    # Binarise the categories
    binarised = preprocessing.LabelBinarizer().fit_transform(df)

    # Format the binarised items into 11 columns
    binarised_basic_4y_list = []
    binarised_basic_6y_list = []
    binarised_basic_9y_list = []
    binarised_high_school_list = []
    binarised_illiterate_list = []
    binarised_professional_course_list = []
    binarised_university_degree_list = []
    binarised_unknown_list = []
    for i in range(len(binarised)):
        item = binarised[i]
        binarised_basic_4y_list.append(item[0])
        binarised_basic_6y_list.append(item[1])
        binarised_basic_9y_list.append(item[2])
        binarised_high_school_list.append(item[3])
        binarised_illiterate_list.append(item[4])
        binarised_professional_course_list.append(item[5])
        binarised_university_degree_list.append(item[6])
        binarised_unknown_list.append(item[7])

    # Return the results as a dictionary
    return {'basic.4y': binarised_basic_4y_list,
            'basic.6y': binarised_basic_6y_list,
            'basic.9y': binarised_basic_9y_list,
            'high.school': binarised_high_school_list,
            'illiterate': binarised_illiterate_list,
            'professional.course': binarised_professional_course_list,
            'university.degree': binarised_university_degree_list,
            'unknown': binarised_unknown_list}


def binarise_job(df):
    # Isolate the values from the data frame
    df = df.values

    # Binarise the categories
    binarised = preprocessing.LabelBinarizer().fit_transform(df)

    # Format the binarised items into 11 columns
    binarised_admin_list = []
    binarised_blue_collar_list = []
    binarised_entrepreneur_list = []
    binarised_housemaid_list = []
    binarised_management_list = []
    binarised_retired_list = []
    binarised_self_employed_list = []
    binarised_services_list = []
    binarised_student_list = []
    binarised_technician_list = []
    binarised_unemployed_list = []
    for i in range(len(binarised)):
        item = binarised[i]
        binarised_admin_list.append(item[0])
        binarised_blue_collar_list.append(item[1])
        binarised_entrepreneur_list.append(item[2])
        binarised_housemaid_list.append(item[3])
        binarised_management_list.append(item[4])
        binarised_retired_list.append(item[5])
        binarised_self_employed_list.append(item[6])
        binarised_services_list.append(item[7])
        binarised_student_list.append(item[8])
        binarised_technician_list.append(item[9])
        binarised_unemployed_list.append(item[10])

    # Return the results as a dictionary
    return {'admin': binarised_admin_list,
            'blue-collar': binarised_blue_collar_list,
            'entrepreneur': binarised_entrepreneur_list,
            'housemaid': binarised_housemaid_list,
            'management': binarised_management_list,
            'retired': binarised_retired_list,
            'self-employed': binarised_self_employed_list,
            'services': binarised_services_list,
            'student': binarised_student_list,
            'technician': binarised_technician_list,
            'unemployed': binarised_unemployed_list}


def binarise_day(df):
    # Isolate the values from the data frame
    df = df.values

    # Binarise the categories
    binarised = preprocessing.LabelBinarizer().fit_transform(df)

    # Format the binarised items into 5 columns
    binarised_monday_list = []
    binarised_tuesday_list = []
    binarised_wednesday_list = []
    binarised_thursday_list = []
    binarised_friday_list = []
    for i in range(len(binarised)):
        item = binarised[i]
        binarised_friday_list.append(item[0])
        binarised_monday_list.append(item[1])
        binarised_thursday_list.append(item[2])
        binarised_tuesday_list.append(item[3])
        binarised_wednesday_list.append(item[4])

    # Return the results as a dictionary
    return {'monday': binarised_monday_list,
            'tuesday': binarised_tuesday_list,
            'wednesday': binarised_wednesday_list,
            'thursday': binarised_thursday_list,
            'friday': binarised_friday_list}


def binarise_month(df):
    # Isolate the values from the data frame
    df = df.values

    # Binarise the categories
    binarised = preprocessing.LabelBinarizer().fit_transform(df)

    # Format the binarised items into 10 columns
    binarised_march_list = []
    binarised_april_list = []
    binarised_may_list = []
    binarised_june_list = []
    binarised_july_list = []
    binarised_august_list = []
    binarised_september_list = []
    binarised_october_list = []
    binarised_november_list = []
    binarised_december_list = []
    for i in range(len(binarised)):
        item = binarised[i]
        binarised_april_list.append(item[0])
        binarised_august_list.append(item[1])
        binarised_december_list.append(item[2])
        binarised_july_list.append(item[3])
        binarised_june_list.append(item[4])
        binarised_march_list.append(item[5])
        binarised_may_list.append(item[6])
        binarised_november_list.append(item[7])
        binarised_october_list.append(item[8])
        binarised_september_list.append(item[9])

    # Return the results as a dictionary
    return {'march': binarised_march_list,
            'april': binarised_april_list,
            'may': binarised_may_list,
            'june': binarised_june_list,
            'july': binarised_july_list,
            'august': binarised_august_list,
            'september': binarised_september_list,
            'october': binarised_october_list,
            'november': binarised_november_list,
            'december': binarised_december_list}


def write_to_xls(df):
    # Write and save the data to an excel document
    writer = pd.ExcelWriter('output.xls', engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()