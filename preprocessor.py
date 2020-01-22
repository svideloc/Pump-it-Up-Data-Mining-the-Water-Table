import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders import LeaveOneOutEncoder



def DataCleaner(values_df, labels_df, test_df):

    # Training Set
    df= pd.merge(values_df, labels_df, on = 'id' )

    #Fills in the mod
    for col in df.columns[df.isna().sum() > 0]:
        mode = df[col].mode()[0]
        df[col].fillna(value = mode, inplace = True)
        
    #dropping    
    to_drop = ['funder', 'num_private', 'subvillage', 'region_code', 'recorded_by', 'source_type', 'waterpoint_type', 'scheme_name', 'payment_type', 'quantity_group']
    df.drop(columns = to_drop, inplace = True)
    #targets to 0,1,2
    df['status_group'] = df['status_group'].map({'functional': 2, 'functional needs repair': 1, 'non functional': 0})

    #date column
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['year_recorded'] = df['date_recorded'].dt.year
    df['month_recorded'] = df['date_recorded'].dt.month
    df.drop(columns = 'date_recorded', inplace = True)

    #Test Set
    #TEST SET TRANSFORM
    test_df = pd.read_csv('test_set_values.csv')

    #Fills in the mod
    for col in test_df.columns[test_df.isna().sum() > 0]:
        mode = test_df[col].mode()[0]
        test_df[col].fillna(value = mode, inplace = True)
        
    #dropping    
    to_drop = ['funder', 'num_private', 'subvillage', 'region_code', 'recorded_by', 'source_type', 'waterpoint_type', 'scheme_name', 'payment_type', 'quantity_group']
    test_df.drop(columns = to_drop, inplace = True)

    #date column
    test_df['date_recorded'] = pd.to_datetime(test_df['date_recorded'])
    test_df['year_recorded'] = test_df['date_recorded'].dt.year
    test_df['month_recorded'] = test_df['date_recorded'].dt.month
    test_df.drop(columns = 'date_recorded', inplace = True)


    #target encode
    target = 'status_group'
    lst_te = ['wpt_name', 'basin', 'region', 'district_code', 'lga', 'ward', 'scheme_management','installer','source']

    #encoder = TargetEncoder()
    encoder = LeaveOneOutEncoder()
    
    te_everything = ['wpt_name', 'basin', 'region', 'district_code', 'lga', 'ward', 'scheme_management','installer','source',
                    'extraction_type', 'extraction_type_group', 'extraction_type_class','management', 'payment', 'water_quality', 'management_group', 'quality_group', 
        'quantity','source_class', 'waterpoint_type_group'] 

    for c in te_everything:
        df[str(c) + '_encoded'] = encoder.fit_transform(df[c].values, df[target]) # TRAINING SET
        test_df[str(c) + '_encoded'] = encoder.transform(test_df[c].values) # TEST SET
        df.drop(columns=c, inplace=True) # TRAINING SET
        test_df.drop(columns=c, inplace=True) # TEST SET
        
#     #one hot encode
#     encoder_ohe = OneHotEncoder(sparse=False)

    ohe = ['extraction_type', 'extraction_type_group', 'extraction_type_class','management', 'payment', 'water_quality', 'management_group', 'quality_group', 
        'quantity','source_class', 'waterpoint_type_group']
    



#     #ONE HOT ENCODING TRAINING SET
#     df_new = df[ohe]
#     encoder_ohe.fit(df_new)
#     x = encoder_ohe.transform(df_new)
#     df1 = pd.DataFrame(x)
#     df = pd.concat([df, df1], axis=1)
#     df.drop(columns=ohe, inplace=True)

#     #ONE HOT ENCODING TEST SET
#     df_new1 = test_df[ohe]
#     x1 = encoder_ohe.transform(df_new1)
#     df2 = pd.DataFrame(x1)
#     test_df = pd.concat([test_df, df2], axis = 1)
#     test_df.drop(columns=ohe, inplace=True)

    return df, test_df