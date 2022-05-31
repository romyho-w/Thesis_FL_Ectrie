import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def import_data(cleveland, switzerland, va, hungarian):
    # Step 2: load the data into pandas DataFrames and then merge the four files into a single dataframe
    cleveland_df = pd.read_csv(cleveland, header=None, na_values =["?", -9.0])
    switzerland_df = pd.read_csv(switzerland, header=None, na_values =["?", -9.0])
    va_df = pd.read_csv(va, header=None, na_values =["?", -9.0])
    hungarian_df = pd.read_csv(hungarian, sep=" ", header=None, na_values =["?", -9.0])

    # add headers to the data frames
    headers = {0 : "Age",
            1 : "Sex", # 1 = male; 0 = female
            2 : "ChestPainType",  # chest pain type, 
                        # Value 1: typical angina, 
                        # Value 2: atypical angina, 
                        # Value 3: non-anginal pain
                        # Value 4: asymptomatic
            3 : "RestingBP", # resting blood pressure 
                            #(in mm Hg on admission to the hospital)
            4 : "Cholesterol", # serum cholestoral in mg/dl
            5 : "FastingBS", # fasting blood sugar > 120 mg/dl 
                        # (1 = true; 0 = false)
            6 : "RestingECG",#  resting electrocardiographic results
                            # Value 0: normal
                            # Value 1: having ST-T wave abnormality 
                            # (T wave inversions and/or ST elevation or depression of > 0.05 mV)
                            # Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
            7 : "MaxHR", #maximum heart rate achieved
            8 : "ExerciseAngina",# exercise induced angina (1 = yes; 0 = no)
            9 : "Oldpeak", # ST depression induced by exercise relative to rest
            10 : "ST_Slope", # the slope of the peak exercise ST segment
                            # Value 1: upsloping
                            # Value 2: flat
                            # Value 3: downsloping
            11 : "ca", # number of major vessels (0-3) colored by flourosopy    
            12 : "thal", # Value 3: normal
                            # Value 6: fixed defect
                            # Value 7: reversable defect
            13 : "HeartDisease" # diagnosis of heart disease (angiographic disease status)
                                # Value 0 = < 50% diameter narrowing,
                                # Value 1 = > 50% diameter narrowing 
                                # It takes 5 levels based on angiographic disease status.
                                # 0-Healthy, 1-diagnosed with stage 1, 2-diagnosed with stage 2, 
                                # 3-diagnosed with stage 3, 4-diagnosed with stage 4.
            }
    cleveland_df = cleveland_df.rename(columns=headers)
    switzerland_df = switzerland_df.rename(columns=headers)
    va_df = va_df.rename(columns=headers)
    hungarian_df = hungarian_df.rename(columns=headers)
    # cleveland_df['Location'] = 'Cleveland'
    # switzerland_df['Location'] = 'Switzerland'
    # va_df['Location']='VA'
    # hungarian_df['Location']='Hungarian'
    hungarian_df = hungarian_df[:-1]
    
    return cleveland_df, switzerland_df, va_df, hungarian_df

    
def new_df(df):
    start_len = df.shape[0]
    # print(start_len)
    # df = pd.concat([cleveland_df, switzerland_df, va_df, hungarian_df])
    df.HeartDisease = df.HeartDisease.replace([1, 2, 3, 4], 1)
    df = df.drop(columns=['thal', 'ca'])
    df = df.drop_duplicates()
    drop_dup= df.shape[0]
    # print how many rows are droped
    # print('drop duplicates', start_len-drop_dup)

    

    # DROP ROWS WITH MORE THAN 35% MISSING VALUES
    perc = 35.0 
    min_count =  int(((100-perc)/100)*df.shape[1] + 1)
    df = df.dropna( axis=0, thresh=min_count)
    drop_mis = df.shape[0]
    # print how many rows are droped
    # print('drop rows with missing values', drop_dup-drop_mis)
    

    # FILL MISSING VALUES WITH MEDIAN
    df.ST_Slope = df.ST_Slope.fillna(df.ST_Slope.median())
    df.RestingBP = df.RestingBP.fillna(df.RestingBP.median())
    df.FastingBS = df.FastingBS.fillna(df.FastingBS.median())
    df.RestingECG = df.RestingECG.fillna(df.RestingECG.median())
    df.Cholesterol = df.Cholesterol.fillna(df.Cholesterol.median())
    df.MaxHR = df.MaxHR.fillna(df.MaxHR.median())
    df.ExerciseAngina = df.ExerciseAngina.fillna(df.ExerciseAngina.median())
    df.Oldpeak = df.Oldpeak.fillna(df.Oldpeak.median())

    # DROP ROW WHERE RESTING BP = 0
    # df = df[df.RestingBP != 0]
    df.RestingBP = df.RestingBP.replace(0, df.RestingBP.median())
    dropBP = df.shape[0]
    # print('drop rows with resting BP', drop_mis-dropBP)
    # # REPLACE CHOLESTROL VALUE = 0 WITH MEDIAN
    # df.Cholesterol = df.Cholesterol.replace(0, df.Cholesterol.median())
    return df
   

def make_dummies(df, cat_feat):
    possible_values = dict()
    for i in cat_feat:
        possible_values[i] = df[i].unique()

    for feature in cat_feat:
        list_of_possible_values = [feature + '_' + str(value) for value in possible_values[feature]]
        dummies = pd.get_dummies(df[feature], prefix = feature).T.reindex(list_of_possible_values).T.fillna(0)
        df = pd.concat([df, dummies], axis=1)

    df = df.drop(cat_feat, axis = 1)

    return df 

def data_prep(X,y):

    def scale_variables(X):
        scaler = StandardScaler()
        scaler.fit(X)
        X_new = scaler.transform(X)
        return X_new

    X = scale_variables(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train = torch.tensor(X_train).float()
    y_train = torch.Tensor(y_train).unsqueeze(1)
    x_test = torch.tensor(X_test).float()
    y_test = torch.Tensor(y_test.reset_index(drop=True)).float().unsqueeze(1)
    return X_train, X_test, y_train, y_test