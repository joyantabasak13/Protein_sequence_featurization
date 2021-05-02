# Protein Sequence Segmentation to Windowsized Segments
Run: Rscript Segmenter.R
Uses Src: CutSeq.R
Input: sequence.csv
Input file Format: 
	.csv file with following named Columns MUST be present. 
	All rows must represent a Succinylated datapoint.

| PLMD.ID  	| Position    	| Sequence    	|
|---------	|--------------	|--------------	|


Output: ID: Internally Assigned
	Class: suc/nonsuc
	Sequence: Windowsized sequence segments with lysine ("k") in the midddle. Allows mirroring.
	ACC_ID: PLMD.ID 

| Id  	| Class    	| Sequence    	| ACC_ID 	|
|-------|--------------	|--------------	|-----------	|

			
NOTE: 	Will require changing working directory in Segmenter.R file
