# BOG_Anomaly-Detection

**Summary:**

As a graduate data science consultant working for Blue Ocean Gear Company (a company that create high-tech buoys that can track gear in the marine environment), the goal of this project is to predict the locations of drifting buoys to ensure faster recovery of lost gear and conduct anomaly detection on the movement of buoy sensors. I started with the descriptive analysis to impute missing values, identify outliers and  conduct feature engineering. In order to obtain more trajectory data for each of our clients’ buoys, I ran thousands of simulations of secured buoys to generate synthetic data using OceanParcels. Then I used simulation data to train the model using LSTM neural network, and then fine-tune models on actual Buoy trajectory data (generate prediction interval where buoys may be located) . After prediction buoy locations using LSTM, I extracted relevant features from the residuals for anomaly detection (buoys typically move in a circular motion, but sometimes they drift). I applied PCA to the standardized features and apply K-Means clustering to identify potential outliers. 

**EDA:**

Our main goals in this milestone are to: (1) clean the dataset by identifying and removing invalid data points; (2) calculate buoys’ average range of motion by fishery; (3) contextualize buoy movement by documenting known traits (e.g., wave and current patterns) of the surrounding water bodies; (4) investigate potential breakaway events; and (5) describe how buoys’ reported metrics, like battery temperature, water temperature, and salinity, vary across time and space.  These tasks will help us prepare, and develop an intuition for, the dataset ahead of predictive modeling. 



**Generate Synthetic data - simulation：**

1. Why do we need it? 
To predict the next location where the buoy would “drift” in the ocean, we need historical data to train the model about the buoy’s past drifting trajectory

2. It would have been ideal to have years of “trajectory data” for each of our client’s buoys

3. Trajectory data simply means if we track a given buoy which is deployed in the water over several months, what are the latitude and longitude coordinates that it would drift to, based on the ocean currents and ocean temperature

4. Trajectory data for the buoys is available only for ~2 years which is insufficient for the machine learning model to learn the pattern of a buoy’s movement

5. Therefore, we simulate the ocean environment using Copernicus data to artificially increase the number of training samples
Copernicus dataset that was previously discussed is used for the purpose of generating synthetic data which we queried using their service for sea surface current velocities (horizontal/eastward/"U" and vertical/northward/"V") that had been created from oceanic general circulation, wave, and tidal models

**Methodology:**

1. We use a Python package OceanParcels (MIT license) which simulates the movement of particles in the ocean over time by using ocean current velocities and temperature as input data.
   
2. We simulate 500 particles in the ocean for every fishery. Each particle is intended to resemble the movement of 1 buoy at different time stamps, thereby generating a particle’s trajectory. A trajectory ID uniquely identifies the multiple locations at which a single particle has “moved” as a result of the ocean’s current velocities and horizontal ocean movement.


**LSTM:**

Goal: Predict the “next sequence of locations” where a particles would drift to, given the previous trajectory of a particle
In our case, we input a particle (i.e. a trajectory ID) and experiment with a 2 layer LSTM model using Keras package in Python

Model Input:
A batch of trajectories with each trajectory consisting of trajectory ID, time, longitude, latitude and temperature up to 100 time steps of historical trajectory data for each of the particles in that batch

Model Output:
For each particle in the batch, the model spits out the next 10 locations where that particle (or the buoy) would drift to - array 

Loss Function:
The function used to define these errors is Mean Squared Error i.e. the model will aim to minimize the mean of squared errors that occur on each particle in the training data 
