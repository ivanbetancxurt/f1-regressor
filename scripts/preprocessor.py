from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

numerical_features = ['grid_pos', 'fastest_lap', 'fastest_lap_avg_speed'] # features to be scaled

preprocessor = ColumnTransformer(
    transformers=[ # add a standard scalar transformer to the numerical features above
        ('scalar', StandardScaler(), numerical_features)
    ],
    remainder='passthrough' # leave the remaining features as they are
)