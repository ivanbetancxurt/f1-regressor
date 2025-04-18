import pandas as pd

def fastest_lap_to_seconds(df): # convert 'mm:ss.ss' format to seconds
    def convert(time): # define conversion funciton
        if time != '300':
            try:
                minutes, seconds = time.split(':')
                return (float(minutes) * 60) + float(seconds)
            except Exception as e:
                print(time)
        else:
            return 300

    df['fastest_lap'] = df['fastest_lap'].apply(convert) # apply function

    return df

def one_hot_encode(df): # one-hot encode categorical columns
    df = pd.get_dummies(df, columns=['circuit_id', 'driver_id', 'constructor_id', 'status'], prefix={ # encode
        'circuit_id': 'circuit',
        'driver_id': 'driver',
        'constructor_id': 'constructor',
        'status': 'status'
    })

    # convert values encoded to True and False to 1 and 0
    bool_cols = df.select_dtypes(include=['bool']).columns 
    df[bool_cols] = df[bool_cols].astype(int)

    df.to_csv('data/dataframes/raw_numerical.csv', index=False) # save

raw_df = pd.read_csv('data/dataframes/raw.csv')
raw_df = fastest_lap_to_seconds(raw_df)
one_hot_encode(raw_df)