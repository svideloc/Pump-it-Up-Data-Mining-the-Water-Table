def DataCleaner(train_set_values_df, train_set_labels_df):


	df= pd.merge(values_df, labels_df, on = 'id' )

    #Fills in the mod
    for col in df.columns[df.isna().sum() > 0]:
        mode = df[col].mode()[0]
        df[col].fillna(value = mode, inplace = True)
        
    #dropping    
    to_drop = ['id','funder', 'num_private', 'subvillage', 'region_code', 'recorded_by', 'source_type', 'waterpoint_type', 'scheme_name', 'payment_type', 'quantity_group']
    df.drop(columns = to_drop, inplace = True)

    #targets to 0,1,2
    df['status_group'] = df['status_group'].map({'functional': 2, 'functional needs repair': 1, 'non functional': 0})

    #date column
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['year_recorded'] = df['date_recorded'].dt.year
    df['month_recorded'] = df['date_recorded'].dt.month
    df.drop(columns = 'date_recorded', inplace = True)

    # df.to_csv('cleaned_columns_dropped.csv')

    #target encode
    target = 'status_group'
    lst_te = ['wpt_name', 'basin', 'region', 'district_code', 'lga', 'ward', 'scheme_management','installer','source']

    encoder = TargetEncoder()

    for c in lst_te:
        df[str(c) + '_encoded'] = encoder.fit_transform(df[c].values, df[target])
        df.drop(columns=c, inplace=True)
        
    #one hot encode
    encoder_ohe = OneHotEncoder(sparse=False)

    ohe = ['extraction_type', 'extraction_type_group', 'extraction_type_class','management', 'payment', 'water_quality', 'management_group', 'quality_group', 
        'quantity','source_class', 'waterpoint_type_group']

    df_new = df[ohe]
    encoder_ohe.fit(df_new)
    x = encoder_ohe.transform(df_new)
    df1 = pd.DataFrame(x)
    df = pd.concat([df, df1], axis=1)
    df.drop(columns=ohe, inplace=True)

    return df
