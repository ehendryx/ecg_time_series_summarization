library(R.matlab)
library(abind)

############################## NOTES ###########################################
# $info[[1]] - Patient ID
# $info[[2]] - Time Matrix
# $info[[3]] - Leads
# $info[[4]] - Annotations
# $info[[5]] - Lead Labels
# $info[[3]][[1]] - Lead 1
# $info[[3]][[1]][[1]] - Lead 1, Data Matrix 1




############################ INCART12 DATA PROCESSING ##########################
# Directory variables
train_dir = "/home/efogarty/Documents/Data Sets/INCART 12/Train"
test_dir = "/home/efogarty/Documents/Data Sets/INCART 12/Test"

# Get a list of .mat files in the directories
train_files = list.files(train_dir, pattern = "\\.mat$")
test_files = list.files(test_dir, pattern = "\\.mat$")




############################# Initializing Lists ###############################
# x lists are the training data, y lists are the annotations, id lists are patient IDs
# training lists
x.train = array(dim = c(0, 150, 12))
y.train = list()
id.train = array(dim = c(0))

# test lists
x.test = array(dim = c(0, 150, 12))
y.test = list()
id.test = array(dim = c(0))




########################## Data Layering Functions #############################
data_layer = function(lead_data){
  # Make an array with desired dimensions to put the lead data in
  layered_array = array(dim = c(nrow(lead_data[[1]]), ncol(lead_data[[1]]), 12))
  # For each lead, add it on the last dimension of the layered_array
  for (i in 1:12) {
    layered_array[,,i] = as.matrix(lead_data[[i]])
  }
  
  return(layered_array)
}




######################### Read Files From Train Folder #########################
for (file in train_files){
  # Read in the file data
  data = readMat(paste0(train_dir, "/", file))
  
  # Matrix for all 12 leads
  # Set recursive to false to prevent unwanted lead concatenation
  leads = unlist(data$info[[3]], recursive = FALSE)
  # Call layering function for the files lead data
  final_array1 = data_layer(leads)
  # Bind the layered lead data to the main x.train array along axis 1 (rows)
  x.train = abind(x.train, final_array1, along = 1)

  # Extract annotations from file data
  annotations = data$info[[4]]
  # Unlist the annotations, then transpose the array so annotations are by row
  annotations = t(unlist(annotations))
  # Append current file's annotations to the full y.train array
  y.train = append(y.train, annotations)
  
  # Extract patient IDs from file data
  tempID.array = array(dim = c(nrow(leads[[1]])), data = unlist(data$info[[1]]))
  # Append current patient ID to the full ID list
  id.train = abind(id.train, tempID.array, along = 1)
}
# Unlist the annotations one last time to concat them all into a single array
y.train = unlist(y.train)
x.train = unname(x.train)
id.train = unname(id.train)




######################### Read Files From Test Folder ##########################
for (file in test_files){
  # Read in the file data
  data = readMat(paste0(test_dir, "/", file))
  
  # Matrix for all 12 leads
  # Set recursive to false to prevent unwanted lead concatenation
  leads = unlist(data$info[[3]], recursive = FALSE)
  # Call layering function for the files lead data
  final_array1 = data_layer(leads)
  # Bind the layered lead data to the main x.train array along axis 1 (rows)
  x.test = abind(x.test, final_array1, along = 1)
  
  # Matrix for annotations
  annotations = data$info[[4]]
  # Unlist the annotations, then transpose the array so annotations are by row
  annotations = t(unlist(annotations))
  # Append current file's annotations to the full y.train array
  y.test = append(y.test, annotations)
  
  # Extract patient IDs from file data
  tempID.array = array(dim = c(nrow(leads[[1]])), data = unlist(data$info[[1]]))
  # Append current patient ID to the full ID list
  id.test = abind(id.test, tempID.array, along = 1)
}
# Unlist the annotations one last time to concat them all into a single array
y.test = unlist(y.test)
id.test = unname(id.test)
x.test = unname(x.test)




############################ Save to .RData file ###############################
save(x.train, x.test, y.train, y.test, id.train, id.test,
     file = "Intermediate Processed INCART Data.RData")

