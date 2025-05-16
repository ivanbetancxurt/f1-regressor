# F1 Regressor

An exploratory machine learning project aimed towards gaining more experience with ML tooling through finding the best predictors for the finishing position of Formula 1 drivers. Manually sourcing the data from the now deprecated [Ergast API](https://ergast.com/mrd/), I explored driver specific features like starting grid position, fastest lap times, and average speed during the fastest lap as well as non-driver specific ones like track and constructor. I explored a random forest regressor, a ridge regressor, and a support vector regressor. The SVR was able to predict finishing positions within about **2 positions** with the most important feature being ___ (by permutation feature importance).

## Methodology

**1. Data Collection**

Formula 1 changes drivers and teams freqently and I wanted to avoid, as much as possible, introducing bias towards drivers and teams that appear in more races in the data.
