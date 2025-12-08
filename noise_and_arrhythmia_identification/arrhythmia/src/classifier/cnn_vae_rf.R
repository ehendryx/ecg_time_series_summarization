# CNN-VAE Random Forest Classifier
setwd("D:/Documents/UCO/Research/R Scripts")
library(randomForest)
library(dplyr) # Needed for %>% function



load("D:/Documents/UCO/Research/Data Sets/INCART12/combined_data.RData")

# Uncomment the data to be classified
#load("D:/Documents/UCO/Research/Data Sets/INCART12/12lead_1D_600mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/12lead_1D_50mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/vcg_1D_150mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/vcg_1D_300mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/lead2_1D_50mv_5layer.RData")
load("D:/Documents/UCO/Research/Data Sets/INCART12/lead2_1D_100mv_5layer.RData")




#################################### RANDOM FOREST #############################
# Training and predicting with Means/Variances
# Setting the seed of the random number generator to make forest generation reproducible
set.seed(3)


# Combine the data and labels into a data frame
data = data.frame(ds2.train.latent, label = factor(ds2.y.train.aami))
# Train the Random Forest
rf_model = randomForest(label ~ ., data = data, ntree = 500, type = "classification")

# Test Random Forest
rf_test_data = data.frame(ds2.test.latent)
rf_predictions = rf_model %>% predict(newdata = rf_test_data)

# Confusion matrix of Random Forest predictions and actual labels
rf_results = table(Actual = ds2.y.test.aami, Predicted = rf_predictions)
print(rf_results)


  
  