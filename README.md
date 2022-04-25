# Card-demand-forecasting
Car Rental Forecasting , ABC is car rental company and they want to forecast the demand hourly.  The Historical hourly data of car demand  is given from 2018-08-18 to 2021-02-28 and task is to predict hourly car demand from 2021- 03-01 to 2022-03-28. 

Demand Forecasting

Can you forecast the demand of the car rentals on an hourly basis?


Problem Statement

ABC is a car rental company based out of Bangalore. It rents cars for both in and out stations at affordable prices. The users can rent different types of cars like Sedans, Hatchbacks, SUVs and MUVs, Minivans and so on.

In recent times, the demand for cars is on the rise. As a result, the company would like to tackle the problem of supply and demand. The ultimate goal of the company is to strike the balance between the supply and demand inorder to meet the user expectations. 

The company has collected the details of each rental. Based on the past data, the company would like to forecast the demand of car rentals on an hourly basis. 


Objective

The main objective of the problem is to develop the machine learning approach to forecast the demand of car rentals on an hourly basis.


Data Dictionary

You are provided with 3 files - train.csv, test.csv and sample_submission.csv

Training set

train.csv contains the hourly demand of car rentals from August 2018 to February 2021.


Variable Description

date.   Date (yyyy-mm-dd)

hour.   Hour of the day

demand. No. of car rentals in a hour


Test set

test.csv contains only 2 variables: date and hour. You need to predict the hourly demand of car rentals for the next 1 year i.e. from March 2021 to March 2022.


Variable      Description

date          Date (yyyy-mm-dd)

hour          Hour of the day


Submission File Format

sample_submission.csv contains 3 variables - date, hour and demand


Variable   Description

date       Date (yyyy-mm-dd)

hour       Hour of the day

demand     No. of car rentals in a hour


Evaluation metric

The evaluation metric for this hackathon is RMSE score.

