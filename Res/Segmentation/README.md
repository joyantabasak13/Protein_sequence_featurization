# Protein Sequence Segmentation to Windowsized Segments
Run: Rscript Segmenter.R <br>
Uses Src: CutSeq.R <br>
Input: sequence.csv <br>
Input file Format: <br>
	.csv file with following named Columns MUST be present.  <br>
	All rows must represent a Succinylated datapoint. <br> 
<br>

| PLMD.ID  	| Position    	| Sequence    	|
|---------	|--------------	|--------------	|

<br>
Output: <br>
 ID: Internally Assigned<br>
	Class: suc/nonsuc <br>
	Sequence: Windowsized sequence segments with lysine ("k") in the midddle. Allows mirroring. <br>
	ACC_ID: PLMD.ID  <br>
<br>

| Id  	| Class    	| Sequence    	| ACC_ID 	|
|-------|--------------	|--------------	|-----------	|

<br>	
<br>		
NOTE: 	Will require changing working directory in Segmenter.R file
