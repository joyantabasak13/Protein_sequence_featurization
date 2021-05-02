CutSeq<-function(acc, SeqString, SuccPosition, WindowSize)
{
  string_split<-strsplit(SeqString, "")[[1]]
  
  Id<-vector('numeric')
  
  Class<-vector('character')
  
  Sequence<-vector('character')
  
  ACC_ID<-vector('character')
  
  x<-1
  
  for (c in string_split)
  {
    if(c == 'K') 
    {
      if(x>WindowSize & x<=length(string_split)-WindowSize)
      {
        s<-substr(SeqString, x-WindowSize, x+WindowSize)
        
        Id<-c(Id,x)
        
        ACC_ID<-c(ACC_ID,acc)
        
        Sequence<-c(Sequence,s)
        
        if(x %in% SuccPosition)
        {
          Class<-c(Class,"suc")
        }
        else
        {
          Class<-c(Class,"nonsuc")
        }
      }
      else if(x<=WindowSize)
      {
        s1<-substr(SeqString, x+1, x+WindowSize)
        
          s1<-stringi::stri_reverse(s1)
        
          s2<-substr(SeqString, x, x+WindowSize)
        
          s<-paste(s1, s2, sep = "")
          
          Id<-c(Id,x)
          
          ACC_ID<-c(ACC_ID,acc)
          
          Sequence<-c(Sequence,s)
          
          if(x %in% SuccPosition)
          {
            Class<-c(Class,"suc")
          }
          else
          {
            Class<-c(Class,"nonsuc")
          }
      }
      else
      {
        s1<-substr(SeqString, x-WindowSize, x)
        
        s2<-substr(SeqString, x-WindowSize, x-1)
        
        s2<-stringi::stri_reverse(s2)
        
        s<-paste(s1, s2, sep = "")
        
        Id<-c(Id,x)
        
        ACC_ID<-c(ACC_ID,acc)
        
        Sequence<-c(Sequence,s)
        
        if(x %in% SuccPosition)
        {
          Class<-c(Class,"suc")
        }
        else
        {
          Class<-c(Class,"nonsuc")
        }
      }
    }
    x = x+1
  }
  df<-data.frame(Id, Class, Sequence, ACC_ID)
  
  return(df)
}