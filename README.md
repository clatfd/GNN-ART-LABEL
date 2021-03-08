# GNN-ART-LABEL
Code and dataset for the MICCAI 2020 paper: Automated Intracranial Artery Labeling using a Graph Neural Network and Hierarchical Refinement 
Paper: https://arxiv.org/abs/2007.14472

# Code
Please have a look at the ArtLabel.ipynb notebook, which contains all the code you need.

# Data
The 729 scans come from five datasets: Anzhen, ArizonaCheck, BRAVE, CROPCheck, Parkinson2TPCheck
Each graph is constructed from one set of intracranial artery traces (from iCafe) and saved in the pickle format under graph/graphsim/dataset_name
Training/validation/test set separation used in our paper is under db.list under each folder

# iCafe
Artery traces are generated from TOF MRA images using iCafe, a semi-automated tool to trace intracranial arteires. Please see the iCafe website for more information
http://icafe.clatfd.cn/

# Own data
If you are using your own artery traces, please construct the graph accordingly to the format we used in the pickle files before running our code.
Please refer to the notebook of "Generate graph.ipynb" for more information about the data format and the code for generating graphs. 
Please make modifications accordingly based on your own data. 
