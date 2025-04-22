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
![image](https://github.com/user-attachments/assets/98928515-5ead-4cac-813d-272570285064)
![image](https://github.com/user-attachments/assets/25287d18-9ce0-4c07-b4c9-5dc3d1d980c3)
![image](https://github.com/user-attachments/assets/9be0f70e-a5d7-4438-bf56-840e8f741240)

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

<!--
## Performance Comparison on SWaT and WADI
| Model      | Metric    | SWaT Mean | SWaT Std | WADI Mean | WADI Std |
|------------|-----------|-----------|----------|-----------|----------|
| **GDN**    | F1        | 0.5803    | 0.1165   |           |          |
|            | Precision | 0.5205    | 0.2220   |           |          |
|            | Recall    | 0.7159    | 0.0261   |           |          |
|            | Accuracy  | 0.8656    | 0.0481   |           |          |
|            | ROC       | 0.8807    | 0.0092   |           |          |
| **ST-GDN** | F1        | **0.7593**|**0.0709**| 0.2339    | 0.0657   |
|            | Precision | **0.9200**|**0.1490**| 0.9289    | 0.0442   |
|            | Recall    |   0.6573  |**0.0184**| 0.1359    | 0.0436   |
|            | Accuracy  | **0.9471**|**0.0286**| 0.9494    | 0.0021   |
|            | ROC       | **0.8931**|  0.0123  | 0.6760    | 0.0555   |
-->

### Notices:
* The first column in .csv will be regarded as index column. 
* The column sequence in .csv don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)
