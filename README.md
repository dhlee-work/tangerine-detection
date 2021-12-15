# Prediction of Tangerine Production through Tangerine Image Object Detection
Project for THEORY AND APPLICATIONS OF DEEP LEARNING LETURE IN YONSEI UNIV. (IIE7721.01-00)  
Predict production on the farm is important. To solve this problem, We use detection model to count product directly. We make dataset for detecting tangerine on the farm. We trained Mask-RCNN Model for detecting tangerines. It has 0.782 AP with IoU=0.50 for bounding box and 0.781 AP with IoU=0.50 for segmentation In counting tangerines, on the train dataset, Target and Output has 0.903 Pearson’s correlation coefficient and 0.921 correlation on the validation dataset. We expect this approach will help make more accurate production forecasts.
## Pipeline
![piple](/fig/pipline.jpg)

## Requirments
albumentations=='1.1.0'   
torch=='1.10.0'  
torchvision=='0.11.1'  
numpy=='1.21.4'  
opencv-python=='4.5.4'  
pandas=='1.3.4'  
Pillow=='1.3.4'  
pycocotools=='2.0.3'  
matplotlib=='3.5.0'  

## Quick Start
#### Data Preprocessing
A1.preprocessing.py  
A2.preprocessing_make_validset.py  

#### Build&Test Model  
B1.train_model.py  
B2.test_model.py  

#### Evaluation Model  
C1.evaluation-loss.py   
C2.evaluation-AP.py  
C3.evaluation-estimate-farm.py   


## Result
#### Detection
![output](/fig/output.png)  

#### Counting tangerine 
![scatter_plot](/fig/scatter_plot.jpg)  

