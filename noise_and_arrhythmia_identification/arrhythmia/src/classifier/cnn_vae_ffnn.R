# CNN-VAE Feed Forward Neural Net Classifier
setwd("D:/Documents/UCO/Research/R Scripts")
library(keras)
library(dplyr) # Needed for %>% function



load("D:/Documents/UCO/Research/Data Sets/INCART12/combined_data.RData")

# Uncomment the data to be classified
load("D:/Documents/UCO/Research/Data Sets/INCART12/12lead_1D_600mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/12lead_1D_50mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/vcg_1D_150mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/vcg_1D_300mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/lead2_1D_50mv_5layer.RData")
#load("D:/Documents/UCO/Research/Data Sets/INCART12/lead2_1D_100mv_5layer.RData")



# One-Hot encode AAMI class labels
labels_one_hot <- to_categorical(as.integer(ds2.y.train.aami)-1)
print(labels_one_hot[1,])




################################# FFNN Classifier ##############################
# Classifier Creation
mv_classifier = keras_model_sequential()

# Add layers to the model
mv_classifier %>%
  layer_flatten(input_shape = dim(ds2.train.latent)[-1]) %>%
  layer_dense(units = 1200, activation = "relu") %>%
  layer_dense(units = 600, activation = "relu") %>%
  layer_dense(units = 300, activation = "relu") %>%
  layer_dense(units = 280, activation = "relu") %>%
  layer_dense(units = 220, activation = "relu") %>%
  layer_dense(units = 200, activation = "relu") %>%
  layer_dense(units = 180, activation = "relu") %>%
  layer_dense(units = 160, activation = "relu") %>%
  layer_dense(units = 120, activation = "relu") %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 4, activation = "softmax")

# Compile the model
mv_classifier %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

# MLP Classifier Training
mv_classifier %>% fit(ds2.train.latent, labels_one_hot, epochs = 10, batch_size = 128, verbose = 2)



# Testing MLP Classifier
mv_classifier_predict = mv_classifier %>% predict(ds2.test.latent)



# Get the max value for each beat's prediction
predictions = array()
for (beat in 1:nrow(mv_classifier_predict)){
  # Find the index with the maximum probability
  index_max <- which.max(mv_classifier_predict[beat,])
  # Convert index to class label
  predictions[[beat]] <- levels(ds2.y.test.aami)[index_max]
}

# Confusion matrix of predictions and actual labels
results = table(Actual = ds2.y.test.aami, Predicted = predictions)
print(results)





