setwd("/home/joyanta/Desktop/Protein_sequence_featurization/Res/Segmentation")
source('/home/joyanta/Desktop/Protein_sequence_featurization/Res/Segmentation/CutSeq.R');

windowsize = 15
fileName <- "sampleInput.csv"
outputFilePrefix = "sampleOutput"
output <- paste(outputFilePrefix,"_window_",windowsize,".csv")

MyData<-read.csv(fileName, header=TRUE, sep=",")

DataSet<-data.frame(Seq=character(), isSuc=numeric())

SuccPosition<-vector()

IDCol = which( colnames(MyData)=="PLMD.ID" )
PosCol = which( colnames(MyData)=="Position" )
SeqCol = which( colnames(MyData)=="Sequence" )

for(i in 1:nrow(MyData))
{
  if((i<nrow(MyData)) && identical(MyData[i,IDCol], MyData[i+1,IDCol]))
  {
    SuccPosition<-c(SuccPosition,MyData[i,PosCol])
  }
  
  else
  {
    SuccPosition<-c(SuccPosition,MyData[i,PosCol])
    
    df = CutSeq(toString(MyData[i,IDCol]), toString(MyData[i,SeqCol]), SuccPosition, windowsize)
    
    DataSet = rbind(DataSet, df, stringsAsFactors=FALSE)
    
    SuccPosition<-vector()
  }
}
write.csv(DataSet, output, row.names=FALSE)

print("DONE")
