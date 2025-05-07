# ST-GDN: Spatio-Temporal Graph Deviation Network

## Installation
### Requirements
* Python == 3.12.3
* cuda == 12.6
* torch == 2.5.0
* torch-geometric == 2.6.1

## Architecture Enhancement: Temporal Attention in ST-GDN

ST-GDN extends the original GDN by integrating a **temporal self-attention mechanism**, allowing the model to adaptively focus on important time steps in the past when detecting deviations.

The updated representative vector in sensor _i_ at time step _t_, is computed as:
![image](https://github.com/user-attachments/assets/c6f91dd9-06f8-4f94-b77b-1a08987b85cd)


### Run
```
    # using gpu
    bash run.sh <gpu_id> <dataset> <STGDN/GDN>

    # or using cpu
    bash run.sh cpu <dataset> <STGDN/GDN>
```
You can change running parameters in the run.sh.


## Data Preparation
```
# put your dataset under data/ directory with the same structure shown in the data/msl/

data
 |-msl
 | |-list.txt    # the feature names, one feature per line
 | |-train.csv   # training data
 | |-test.csv    # test data
 |-your_dataset
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...

```
SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/)


## Performance Comparison on SWaT and WADI
| Model      | Metric    | SWaT Mean | SWaT Std | WADI Mean | WADI Std |
|------------|-----------|-----------|----------|-----------|----------|
| **GDN**    | F1        | 0.6358    | 0.1535   | 0.4433    | 0.0302   |
|            | Precision | 0.6057    | 0.2653   | 0.7468    | 0.1118   |
|            | Recall    | 0.7310    | 0.0379   | 0.3244    | 0.0655   |
|            | Accuracy  | 0.8809    | 0.0571   | 0.9688    | 0.0007   |
|            | AUC       | 0.8899    | 0.0220   | 0.8120    | 0.0235   |
| **ST-GDN** | F1        | **0.7854**|**0.0800**| **0.4437**|**0.0071**|
|            | Precision | **0.9328**|**0.1338**| **0.8747**|**0.0438**|
|            | Recall    | 0.6881    | 0.0613   | 0.2968    |**0.0040**|
|            | Accuracy  | **0.9494**|**0.0256**| **0.9711**|**0.0006**|
|            | AUC       | **0.9001**|**0.0182**| **0.8137**| 0.0292   |


### Notices:
* The first column in .csv will be regarded as index column. 
* The column sequence in .csv don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)
