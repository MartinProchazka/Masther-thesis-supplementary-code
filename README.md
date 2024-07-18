# Classification in data streams with abrupt concept drift in a subset of features
This dictionary serves as supplementary code for master thesis.

Main directory contain 3 files and one sub-directories.
- **classes.py** represents main implementation of proposed procedures
- **postprocessing_functions.py** represents suplementary function as graph generation or dataset creation.
- **evaluation_source_code.ipynb** represents evaluations presented in Chapter 5 except ones using DREBIN dataset

DREBIN sub-directory represent evaluations on DREBIN dataset. It consists of two sub-directories. The first is **results**, representing output files from according evaluations. The second is **evaluations** and consists of all files used for evaluation. The DREBIN dataset can be obtained at [kaggle](https://www.kaggle.com/datasets/fabriciojoc/fast-furious-malware-data-stream) and needs to be stored in **evaluations** sub-directory or filepath in ..._main.py must be adjusted.

**DREBIN** sub-directory contains three types of files
- **..._main.py** representing specific setting of evaluation
- **..._test.py** representing lauching scrip
- **..._result.out** representing obtained raw data, of according experimens. These data are utilizes in **evaluation_source_code.ipynb**.

Name of the files in **DREBIN** sub-directory  represent used sub-procedure and according downsampling. To be precise:
- DWM... represents Dynamic Weighted Majority algorithm, without any sub-procedure 
- GMM... represents DWM algorithm, with GMM-based sub-procedure
- Hellinger... represents DWM algorithm, with Hellinger-based sub-procedure

downsampling is represented as follows:
- ...d2... represents downsampling on all samples with probability 0.5
- ...d6... represents downsampling on all samples with probability 5/6
- ...db2... represents downsampling on benign samples only with probability 0.5
- ...db6... represents downsampling on benign samples only with probability 5/6


Specificaly, evaluations in sub-directories can be launched by **..._test.py** files, which runs 10 parallely\
**..._main.py** for distinct random numbers.
Example:
```sh
python DWMd2_test.py
```
runs 10 paralelly **DWM_main.py** for various random numbers using 50% random downsampling. There must be **classes.py**, **postprocessing_functions.py** according **..._main.py** and the dataset **drebin_drift.parquet.zip** in the directory where the evaluation is launged. 

The class structure of classes.py is as follows:
There are 3 main classes standing for main procedures of the thesis:
- DWM, class following general outline of our solution and according to initialization performs DWM algorithm with or without the sub-procedures. Uses classes
    - NBModel, class performing Gaussian Naive Bayes classifier with possibility of prediction using restricted feature vector
    - GMM_overal_process and Hellinger_detection if sub procedures are required
- GMM_overal_process, overall process based on GMM algorithm, described in thesis. Uses classes
    - Cluster_mixture, class for single mixture of clusters, containing class
        - single_cluster, class for single cluster
- Hellinger_detection, overall process based on Hellinger drift detection, described in thesis. Uses class
    - Process_single_feature_subse, class, which perform Hellinger drift detection algorithm for single feature subset (given by dependencies)


All evaluation settings and used algorithms are presented in the Thesis: Classification in data streams with abrupt concept drift in a subset of features
