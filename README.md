# F1 Regressor

An exploratory machine learning project aimed towards gaining more experience with ML tooling through finding the best predictors for the finishing position of Formula 1 drivers. Manually sourcing the data from the now deprecated [Ergast API](https://ergast.com/mrd/), I explored driver specific features like starting grid position, fastest lap times, and average speed during the fastest lap as well as non-driver specific ones like track and constructor. I explored a random forest regressor, a ridge regressor, and a support vector regressor. The SVR was able to predict finishing positions within about **2 positions** with the most important feature being ___ (by permutation feature importance).

## Methodology

**1. Data Collection**

Formula 1 changes drivers and teams freqently and I wanted to avoid, as much as possible, introducing a bias towards drivers and teams that appear in more frequently in races (or one against those that don't). This led to the decision of only considering the last 20 seasons (from 2025). I retrieved JSONs from the [Ergast API](https://ergast.com/mrd/) containing the details and results of every race (each JSON representing one race) for each season, stripping away extraneous information and organizing them into a dataframe where every instance holds information about the driver and their results at the specified race. Alongside the target variable (integer finsishing position), each instance in the raw dataframe had the following features:

 * Circuit name (String)
 * Driver name (String)
 * Constructor name (String)
 * Starting grid position (Integer)
 * Status - whether or not the driver finished the race, and if not, why (String)
 * Missing lap flag - whether or not there was a fastest lap recorded for the driver (0 or 1)
 * Driver's fastest lap time if there was a fastest lap recorded for them. Otherwise, the integer 300 representing 300 seconds (String | 300)
 * Average speed during the driver's fastest lap, if it was recorded. Otherwise, the integer 0 representing 0 kph (NumPy float | 0)

 I chose deal with missing fastest lap times like this becuase I felt it was better than not considering lap times at all. The desicion to impute the missing values with 300s and 0kph was somewhat arbitrary, but my goal was to "punish" these instances as a missing fastest lap time means that the driver didn't even start the race. On top of this, I thought it could be fruitful to build a correlation between the missing lap flag feature and the status feature (every time the missing lap flag has a value of 1, the status feature will have a value of something other than "Finished" and vice versa). The resulting table is `raw.csv`.

 **2. Data preprocessing**

 The first preprocessing I did was translating the strings into machine readable values. I converted the average speeds of the fastest laps from a string in "mm:ss.ss" format to a number representing how may seconds that is. It's worth noting that none of these converted values get very close to the 300 second imputation for missing lap times. After that I one hot encoded the circuit names, driver names, constructors, and status, greatly expanding the number of features in the data. 

 Next, I created the first part of my pipeline. As a ColumnTransformer from ScikitLearn, I applied the ScikitLearn standard scaler to all the features that were originally numerical. The other half of the pipeline was the specified model. The resulting table, before applying standard scaling since it's applied at training time, is `raw_numerical.csv`.

 