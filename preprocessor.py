import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import BaggingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder



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

    encoder = TargetEncoder()

    for c in lst_te:
        df[str(c) + '_encoded'] = encoder.fit_transform(df[c].values, df[target]) # TRAINING SET
        test_df[str(c) + '_encoded'] = encoder.transform(test_df[c].values) # TEST SET
        df.drop(columns=c, inplace=True) # TRAINING SET
        test_df.drop(columns=c, inplace=True) # TEST SET
        
    #one hot encode
    encoder_ohe = OneHotEncoder(sparse=False)

    ohe = ['extraction_type', 'extraction_type_group', 'extraction_type_class','management', 'payment', 'water_quality', 'management_group', 'quality_group', 
        'quantity','source_class', 'waterpoint_type_group']


    #ONE HOT ENCODING TRAINING SET
    df_new = df[ohe]
    encoder_ohe.fit(df_new)
    x = encoder_ohe.transform(df_new)
    df1 = pd.DataFrame(x)
    df = pd.concat([df, df1], axis=1)
    df.drop(columns=ohe, inplace=True)

    #ONE HOT ENCODING TEST SET
    df_new1 = test_df[ohe]
    x1 = encoder_ohe.transform(df_new1)
    df2 = pd.DataFrame(x1)
    test_df = pd.concat([test_df, df2], axis = 1)
    test_df.drop(columns=ohe, inplace=True)

    return df, test_df


class CustomPandasTransformer(BaseEstimator, TransformerMixin):
    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame, but got type=%s" 
                            % type(X))
        return X
    
    @staticmethod
    def _validate_columns(X, cols):
        scols = set(X.columns)  # set for O(1) lookup
        if not all(c in scols for c in cols):
            raise ValueError("all columns must be present in X")



class DummyEncoder(CustomPandasTransformer):
    def __init__(self, columns, sep='_', drop_one_level=True, tmp_nan_rep='N/A'):
        self.columns = columns
        self.sep = sep
        self.drop_one_level = drop_one_level
        self.tmp_nan_rep = tmp_nan_rep
        
    def fit(self, X, y=None):
        # validate the input
        X = self._validate_input(X).copy()  # get a copy
        
        # parameter validation happens here:
        tmp_nan = self.tmp_nan_rep
        
        # validate all the columns present
        cols = self.columns
        self._validate_columns(X, cols)
                
        # for each column, fit a label encoder
        lab_encoders = {}
        for col in cols:
            vec = [tmp_nan if pd.isnull(v) 
                   else v for v in X[col].tolist()]
            
            # if the tmp_nan value is not present in vec, make sure it is
            # so the transform won't break down
            svec = list(set(vec))
            if tmp_nan not in svec:
                svec.append(tmp_nan)
            
            le = LabelEncoder()
            lab_encoders[col] = le.fit(svec)
            
            # transform the column, re-assign
            X[col] = le.transform(vec)
        
        # fit a single OHE on the transformed columns - but we need to ensure
        # the N/A tmp_nan vals make it into the OHE or it will break down later.
        # this is a hack - add a row of all transformed nan levels
        ohe_set = X[cols]
        ohe_nan_row = {c: lab_encoders[c].transform([tmp_nan])[0] for c in cols}
        ohe_set = ohe_set.append(ohe_nan_row, ignore_index=True)
        ohe = OneHotEncoder(sparse= False).fit(ohe_set)
        
        # assign fit params
        self.ohe_ = ohe
        self.le_ = lab_encoders
        self.cols_ = cols
        
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'ohe_')
        X = self._validate_input(X).copy()
        
        # fit params that we need
        ohe = self.ohe_
        lenc = self.le_
        cols = self.cols_
        tmp_nan = self.tmp_nan_rep
        sep = self.sep
        drop = self.drop_one_level
        
        # validate the cols and the new X
        self._validate_columns(X, cols)
        col_order = []
        drops = []
        
        for col in cols:
            # get the vec from X, transform its nans if present
            vec = [tmp_nan if pd.isnull(v) 
                   else v for v in X[col].tolist()]
            
            le = lenc[col]
            vec_trans = le.transform(vec)  # str -> int
            X[col] = vec_trans
            
            # get the column names (levels) so we can predict the 
            # order of the output cols
            classes = ["%s%s%s" % (col, sep, clz) for clz in le.classes_.tolist()]
            col_order.extend(classes)
            
            # if we want to drop one, just drop the last
            if drop:
                drops.append(classes[-1])
                
        # now we can get the transformed OHE
        ohe_trans = pd.DataFrame.from_records(data=ohe.transform(X[cols]), 
                                              columns=col_order)
        
        # set the index to be equal to X's for a smooth concat
        ohe_trans.index = X.index
        
        # if we're dropping one level, do so now
        if drops:
            ohe_trans = ohe_trans.drop(drops, axis=1)
        
        # drop the original columns from X
        X = X.drop(cols, axis=1)
        
        # concat the new columns
        X = pd.concat([X, ohe_trans], axis=1)
        
        return X



class BaggedRegressorImputer(CustomPandasTransformer):
    def __init__(self, impute_cols, base_estimator=None, n_estimators=10, 
                 max_samples=1.0, max_features=1.0, bootstrap=True, 
                 bootstrap_features=False, n_jobs=1,
                 random_state=None, verbose=0):
        
        self.impute_cols = impute_cols
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y=None):
        X = self._validate_input(X)  # don't need a copy this time
        
        # validate the columns
        cols = self.impute_cols
        self._validate_columns(X, cols)
        
        # drop off the columns we'll be imputing as targets
        regressors = {}
        targets = {c: X[c] for c in cols}
        X = X.drop(cols, axis=1)  # these should all be filled in (no NaN)
        
        for k, target in targets.items():
            # split X row-wise into train/test where test is the missing
            # rows in the target
            test_mask = pd.isnull(target)
            train = X.loc[~test_mask]
            train_y = target[~test_mask]
            
            # fit the regressor
            regressors[k] = BaggingRegressor(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                n_jobs=self.n_jobs, 
                random_state=self.random_state,
                verbose=self.verbose, oob_score=False,
                warm_start=False).fit(train, train_y)
            
        # assign fit params
        self.regressors_ = regressors
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'regressors_')
        X = self._validate_input(X).copy()  # need a copy
        
        cols = self.impute_cols
        self._validate_columns(X, cols)
        
        # fill in the missing
        models = self.regressors_
        for k, model in models.items():
            target = X[k]
            
            # split X row-wise into train/test where test is the missing
            # rows in the target
            test_mask = pd.isnull(target)
            
            # if there's nothing missing in the test set for this feature, skip
            if test_mask.sum() == 0:
                continue
            test = X.loc[test_mask].drop(cols, axis=1)  # drop impute cols
            
            # generate predictions
            preds = model.predict(test)
            
            # impute!
            X.loc[test_mask, k] = preds
            
        return X