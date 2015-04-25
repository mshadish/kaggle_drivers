Kaggle: Driver Telematics
================
This project contains several folders related to the Kaggle Driver Telematics anomaly detection competition. Given a set of files representing driving trips, the task is to establish a driver fingerprint and predict, with some confidence, whether or not a particular driving file matches our established driver fingerprint.  
  
Specifics:  
We are given over 2700 folders, and within each folder is 200 CSV files. Each CSV represents a single driving trip taken. Supposedly, each folder of 200 files represents a single driver. However, not all of the 200 files in each folder actually belong to our given driver, and so we must detect some unknown number of anomalies (with some probability) within each folder.  
  
Feature Extraction  
=============  
feature_extraction.R  

Use this script to turn the data in each folder into corresponding tables in CSV format. To run this, simply change the "path" variable in the .R script to match the location of the driver folders.  
  
Exploratory Data Analysis  
=============  
exploratory_data_analysis.R  

Use this script to generate some basic histograms to visualize the distributions of the different features extracted within a given folder. Again, simply change the path variable in the .R script to the location of the data, this time to the '_summary.csv' files generated by the feature extraction process.  Simply pass in a set of (random) numbers to the genBoxplot() function and the name of the column to be analyzed.  
  
Modeling  
=============  
Our methodology is to use supervised modeling techniques in an unsupervised environment.  
  
1) Select a driver and all 200 of the associated trips as our target class  
2) Select some random sample of trips from other drivers and label these as noise  
3) Bootstrap a sample from our target class such that it is balanced with our noise data  
4) Train a supervised model on our pseudo-labeled data  
5) Use this trained model to make probabilistic predictions on trips from our original target class  
  
In our tests, we've found that approximately 2800 data points from each class provides a good balance of data quantity without sacrificing performance.  
  
Results  
=============  
We are evaluated by area-under-the-curve.  Using a combination of a bagged logistic regression ensemble and a random forest as our model, we can achieve an AUC of just over 0.85.  This puts us in the top 30% of over 1500 contestants.