
# ORD KNN for 12 leads ----------------------------------------------------

library(caret)
library(class)
library(foreach)
library(doParallel)

load("/home/efogarty/Documents/Data Sets/INCART 12/combined_data.RData")

# Uncomment the data to be classified
#load("/home/efogarty/Documents/Data Sets/INCART 12/Multi-Lead CNNVAE Extracted Data/12lead_1D_600mv_5layer.RData")
#load("/home/efogarty/Documents/Data Sets/INCART 12/Multi-Lead CNNVAE Extracted Data/12lead_1D_50mv_5layer.RData")
#load("/home/efogarty/Documents/Data Sets/INCART 12/Multi-Lead CNNVAE Extracted Data/vcg_1D_150mv_5layer.RData")
#load("/home/efogarty/Documents/Data Sets/INCART 12/Multi-Lead CNNVAE Extracted Data/vcg_1D_300mv_5layer.RData")
#load("/home/efogarty/Documents/Data Sets/INCART 12/Multi-Lead CNNVAE Extracted Data/lead2_1D_50mv_5layer.RData")
load("/home/efogarty/Documents/Data Sets/INCART 12/Multi-Lead CNNVAE Extracted Data/lead2_1D_100mv_5layer.RData")

set.seed(1)

numCores <- detectCores()
numCores

registerDoParallel(numCores - 4)  # use multicore, set to the number of our cores

train_dat <- data.frame(ds2.train.latent)
train_labels <- ds2.y.train.binary
test_dat <- data.frame(ds2.test.latent)



# Cross Validation --------------------------------------------------------

#trControl <- trainControl(method = "cv", number = 5)
#tune_grid <- expand.grid(k = c(3, 5, 10))



knn_fit = foreach(i=c(3,5,10), .packages='class') %dopar% {
  knn.cv(train_dat, 
         train_labels,
         k = i)
}

confusionMatrix(table(Actual = ds2.y.train.binary, Predicted = knn_fit[[3]]))



# KNN with k = 3 ----------------------------------------------------------

knn_mod <- knn(train = train_dat,
               test = test_dat, 
               cl = train_labels, 
               k = 3)

knn_cm <- table(Actual = ds2.y.test.binary, Predicted = knn_mod)


#save(knn_fit, file= "/home/efogarty/Documents/Grid Search Results/lead2_50mv_knn_cv.RData")
write.csv(knn_cm, file = "/home/efogarty/Documents/Grid Search Results/lead2_100mv_knn.csv")

rm(list=ls())
gc(reset=TRUE)
gc()






