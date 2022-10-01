# antimicrobial-peptides
iAMP-DL: Identifying short antimicrobial peptides using long short-term memory incorporated with convolutional neural networks
+ Train process:
Syntax: python3 train.py

+ Test process: 
Syntax: python3 test.py <FastA input file> <CSV result file>
python3 test.py test_sample.fasta test_sample_result.csv

+ Example of "FastA input file": 
>30_136
ILPIRSLIKKLL
>5_30_768
VNPIILGVLPKFVCLITKKC 
>30_63
AIGPVADLHI
>30_115
APVPGLSPFRVV

+ Example of "CSV result file": 
Name,Sequence,Classification,Probability
>30_136,ILPIRSLIKKLL,AMP,0.8811
>5_30_768,VNPIILGVLPKFVCLITKKC,AMP,0.7944
>30_63,AIGPVADLHI,Non AMP,0.2909
>30_115,APVPGLSPFRVV,Non AMP,0.3453
