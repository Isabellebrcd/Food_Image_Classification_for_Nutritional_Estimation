# AI System - Food Image Classification for Nutritional Estimation



##  Results of the CNN with the first 10 classes 

| Jupyter/Colab | Epochs | Batch_size | K-fold | Optimizer | Loss | Accuracy | Classification Report | Other |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
Jupyter | 50 |  32 | No | Adam | 1.3861 | 0.5795 | Bad | Input size 128*128 |
Colab | 10 |  16 | No | Adam | 2.9226 | 0.4007 | OK | More complex model |
Colab | 100 - EarlyStopping 47 |  64 | No | Adam | 1.4994 | 0.5060 | OK | More complex model & ReduceLROnPlateau |
Colab | 10 | not specified | 5 | Adam | 1.8237 | 0.3225 | Bad | Same model |
Colab | 40 |  not specified | 7 | Adam | 1.5056 | 0.4603 | OK | Same model - ReduceLROnPlateau |
Colab | 20 |  64 | No | Adam | 2.1388 | 0.2196 | Bad | Basic Pre-trained ResNet |
Colab | 20 |  not specified | 5 | Adam | 2.1477 | 0.2072 | Bad | Basic Pre-trained ResNet |
Colab | 70 |  64 | No | Adam | 2.1944 | 0.1853 | Bad | With data Augmentation |
Colab | 40 |  Not specified | 5 | Adam | 2.1609 | 0.2056 | Bad | With data Augmentation |
Colab | 100 - EarlyStopping 21 |  64 | No | Adam | 2.1604 | 0.1513 | Very Bad | Pre-trained ResNet with the more complex model |
Colab | 100 - EarlyStopping 25 |  64 | No | Adam | 1.2128 | 0.6273 | Better | More complex model & ReduceLROnPlateau |
Colab | 45 - EarlyStopping 13 |  64 | No | Adam | 1.4844 | 0.5107 | Better | Same model & Data Augmentation |
Colab | 45 - EarlyStopping 6 |  64 | No | Adam | **1.1662** | **0.6393** | Better | Same model & ReduceLROnPlateau |
Colab | 45 - EarlyStopping 6 |  128 | No | Adam | 2.8044 | 0.1340 | Bad | Same model & ReduceLROnPlateau |
Colab | 45 - EarlyStopping 25 |  64 | No | Adam | **1.2225** | **0.6353** | Better | Same model & ReduceLROnPlateau |
Colab | 100 - EarlyStopping 24 |  64 | No | RMSprop | **1.1955** | **0.6313** | Better | Same model & ReduceLROnPlateau |
Colab | 100 |  64 | No | RMSprop | **1.3335** | **0.6527** | Better | Same model & ReduceLROnPlateau |

