# Data processing for Operation ??? --------------------------------------------
#
# To do:
# - Calculate VCG
# - Create additional train/test splits for both DS1 and DS2
# - Define AAMI classes for Y
# - Remove beats other than "N" from DS1
# - Remove Q class beats from DS2
#
# Resulting data objects to use:
# - CNN VAE
#   - ds1.train.x and ds1.test.x used for single lead and 12-lead models
#   - ds1.train.x.vcg and ds1.test.x.vcg used for VCG models
# - Classification
#   - ds2.train.x and ds2.test.x used for single lead and 12-lead models
#   - ds2.train.x.vcg and ds2.test.x.vcg used for VCG models
#   - ds2.y.train.aami and ds2.y.test.aami used as response for all models



load("Intermediate Processed INCART Data.RData") # divided into x.train, y.train (DS1) and x.test, y.test (DS2)



# Kors matrix ------------------------------------------------------------------

kors <- matrix(c(0.38, -0.07, 0.11,
                 -0.07, 0.93, -0.23,
                 -0.13, 0.06, -0.43,
                 0.05, -0.02, -0.06,
                 -0.01, -0.05, -0.14,
                 0.14, 0.06, -0.2,
                 0.06, -0.17, -0.11,
                 0.54, 0.13, 0.31), nrow = 8, ncol = 3, byrow = TRUE)

leads <- c(1,2,7,8,9,10,11,12) # leads needed to calculate vcg




# VCG for x.train --------------------------------------------------------------

n.x.train <- dim(x.train)[1] 

vcg.x.train<- array(numeric(), dim = c(n.x.train, 150, 3))

for(beat2 in 1:n.x.train){
  tempbeat2 <- x.train[beat2,,leads]%*%kors
  vcg.x.train[beat2,,1] <- tempbeat2[,1]
  vcg.x.train[beat2,,2] <- tempbeat2[,2]
  vcg.x.train[beat2,,3] <- tempbeat2[,3]
}



# VCG for x.test ---------------------------------------------------------------

n.x.test <- dim(x.test)[1] 

vcg.x.test <- array(numeric(), dim = c(n.x.test, 150, 3))

for(beat in 1:n.x.test){
  tempbeat <- x.test[beat,,leads]%*%kors
  vcg.x.test[beat,,1] <- tempbeat[,1]
  vcg.x.test[beat,,2] <- tempbeat[,2]
  vcg.x.test[beat,,3] <- tempbeat[,3]
}



# Data splits ------------------------------------------------------------------
# I02m, I03m, I57m, I58m removed for zero variance beats

ds1.train.id <- c('I04m', 'I05m', 'I08m', 'I09m', 'I10m', 'I11m', 'I12m',
               'I13m', 'I14m', 'I23m', 'I24m', 'I25m', 'I26m', 'I33m', 'I34m')

ds1.test.id <- c('I01m', 'I06m', 'I07m', 'I18m', 'I19m', 'I27m', 'I28m', 
                 'I29m', 'I30m', 'I31m', 'I32m', 'I35m', 'I36m', 'I37m', 'I51m', 
                 'I52m', 'I53m')

ds2.train.id <- c('I16m', 'I17m', 'I20m', 'I21m', 'I22m','I40m', 'I41m', 'I42m', 
                  'I43m', 'I47m', 'I48m', 'I59m', 'I60m', 'I61m', 'I62m', 'I63m', 
                  'I64m', 'I68m', 'I69m', 'I74m', 'I75m')

ds2.test.id <- c('I15m',  'I38m', 'I39m', 'I44m', 'I45m', 'I46m', 'I49m', 'I50m', 
                 'I54m', 'I55m', 'I56m', 'I65m', 'I66m', 'I67m', 'I70m', 'I71m', 
                 'I72m', 'I73m')


# DS1 
ds1.train.x <- x.train[which(id.train %in% ds1.train.id),,]
ds1.test.x <- x.train[which(id.train %in% ds1.test.id),,]
ds1.train.x.vcg <- vcg.x.train[which(id.train %in% ds1.train.id),,]
ds1.test.x.vcg <- vcg.x.train[which(id.train %in% ds1.test.id),,]

ds1.train.y <- y.train[which(id.train %in% ds1.train.id)]
ds1.test.y <- y.train[which(id.train %in% ds1.test.id)]

# DS2 
ds2.train.x <- x.test[which(id.test %in% ds2.train.id),,]
ds2.test.x <- x.test[which(id.test %in% ds2.test.id),,]
ds2.train.x.vcg <- vcg.x.test[which(id.test %in% ds2.train.id),,]
ds2.test.x.vcg <- vcg.x.test[which(id.test %in% ds2.test.id),,]

ds2.train.y <- y.test[which(id.test %in% ds2.train.id)]
ds2.test.y <- y.test[which(id.test %in% ds2.test.id)]



# Convert Y to AAMI classes ----------------------------------------------------

# AAMI Classes:
# N = N, L, R, B, e, j, n
# S = A, a, J, S
# V = V, E
# F = F
# Q = /, f, Q

N1 <- c("N", "L", "R", "B", "e", "j", "n")
S1 <- c("A", "a", "J", "S")
V1 <- c("V", "E")
F1 <- c("F")
Q1 <- c("/", "f", "Q")




# DS1 y.train

ds1.y.train.aami <- NULL

for(i in 1:length(ds1.train.y)){
  if(ds1.train.y[i] %in% N1){
    ds1.y.train.aami[i] <- "N"
  } else if (ds1.train.y[i] %in% S1){
    ds1.y.train.aami[i] <- "S"
  } else if (ds1.train.y[i] %in% V1){
    ds1.y.train.aami[i] <- "V"
  } else if (ds1.train.y[i] %in% F1){
    ds1.y.train.aami[i] <- "F"
  } else if (ds1.train.y[i] %in% Q1){
    ds1.y.train.aami[i] <- "Q"
  }
}


# DS1 y.test

ds1.y.test.aami <- NULL

for(i in 1:length(ds1.test.y)){
  if(ds1.test.y[i] %in% N1){
    ds1.y.test.aami[i] <- "N"
  } else if (ds1.test.y[i] %in% S1){
    ds1.y.test.aami[i] <- "S"
  } else if (ds1.test.y[i] %in% V1){
    ds1.y.test.aami[i] <- "V"
  } else if (ds1.test.y[i] %in% F1){
    ds1.y.test.aami[i] <- "F"
  } else if (ds1.test.y[i] %in% Q1){
    ds1.y.test.aami[i] <- "Q"
  }
}




# DS2 y.train

ds2.y.train.aami <- NULL

for(i in 1:length(ds2.train.y)){
  if(ds2.train.y[i] %in% N1){
    ds2.y.train.aami[i] <- "N"
  } else if (ds2.train.y[i] %in% S1){
    ds2.y.train.aami[i] <- "S"
  } else if (ds2.train.y[i] %in% V1){
    ds2.y.train.aami[i] <- "V"
  } else if (ds2.train.y[i] %in% F1){
    ds2.y.train.aami[i] <- "F"
  } else if (ds2.train.y[i] %in% Q1){
    ds2.y.train.aami[i] <- "Q"
  }
}



# DS2 y.test

ds2.y.test.aami <- NULL

for(i in 1:length(ds2.test.y)){
  if(ds2.test.y[i] %in% N1){
    ds2.y.test.aami[i] <- "N"
  } else if (ds2.test.y[i] %in% S1){
    ds2.y.test.aami[i] <- "S"
  } else if (ds2.test.y[i] %in% V1){
    ds2.y.test.aami[i] <- "V"
  } else if (ds2.test.y[i] %in% F1){
    ds2.y.test.aami[i] <- "F"
  } else if (ds2.test.y[i] %in% Q1){
    ds2.y.test.aami[i] <- "Q"
  }
}



# Remove non-normal beats from DS1 ---------------------------------------------

ds1.train.x <- ds1.train.x[which(ds1.y.train.aami == "N"),,]
ds1.test.x <- ds1.test.x[which(ds1.y.test.aami == "N"),,]
ds1.train.x.vcg <- ds1.train.x.vcg[which(ds1.y.train.aami == "N"),,]
ds1.test.x.vcg <- ds1.test.x.vcg[which(ds1.y.test.aami == "N"),,]



# Remove Q beats from DS2 test (none in train) ---------------------------------

ds2.test.x <- ds2.test.x[-which(ds2.y.test.aami == "Q"),,]
ds2.test.x.vcg <- ds2.test.x.vcg[-which(ds2.y.test.aami == "Q"),,]
ds2.y.test.aami <- as.factor(ds2.y.test.aami[-which(ds2.y.test.aami == "Q")])
ds2.y.test.aami <- relevel(ds2.y.test.aami, ref = "N") 

ds2.y.train.aami <- as.factor(ds2.y.train.aami)
ds2.y.train.aami <- relevel(ds2.y.train.aami, ref = "N")



# Save relevant data objects ----------------------------------------------

save(ds1.train.x, ds1.test.x, ds1.train.x.vcg, ds1.test.x.vcg, 
     ds2.train.x, ds2.test.x, ds2.train.x.vcg, ds2.test.x.vcg,
     ds2.y.train.aami, ds2.y.test.aami, file = "Fully Processed INCART Data.RData")

