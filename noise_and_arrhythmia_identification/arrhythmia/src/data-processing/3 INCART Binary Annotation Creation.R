# INCART 12 - Converting AMII classes to binary 'N' and 'V' classes
load("Fully Processed INCART Data.RData")



# Recoding train annotations to binary classes
ds2.y.train.binary = array()
for (annot in 1:length(ds2.y.train.aami)){
  if (ds2.y.train.aami[annot] == "N"){
    ds2.y.train.binary[annot] = "N"
  }
  else if (ds2.y.train.aami[annot] == "F" || ds2.y.train.aami[annot] == "S" || ds2.y.train.aami[annot] == "V"){
    ds2.y.train.binary[annot] = "V"
  }
}



# Recoding test annotations to binary classes
ds2.y.test.binary = array()
for (annot in 1:length(ds2.y.test.aami)){
  if (ds2.y.test.aami[annot] == "N"){
    ds2.y.test.binary[annot] = "N"
  }
  else if (ds2.y.test.aami[annot] == "F" || ds2.y.test.aami[annot] == "S" || ds2.y.test.aami[annot] == "V"){
    ds2.y.test.binary[annot] = "V"
  }
}

save(ds1.test.x, ds1.train.x,
     ds1.test.x.vcg, ds1.train.x.vcg,
     ds2.test.x, ds2.train.x,
     ds2.test.x.vcg, ds2.train.x.vcg,
     ds2.y.test.aami, ds2.y.train.aami,
     ds2.y.test.binary, ds2.y.train.binary,
     file = "Fully Processed INCART Data.RData")

