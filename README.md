# PSMNET_Pruning(PYTORCH)
Using Network slimming and DoReFa Quantization
## Dependencies
torch v0.3.1, torchvision v0.2.0
## Prune
The dataset argument specifies which dataset to use: Kitti 2015.   
`python prune_slim.py --percent 0.6`  
##  Finetune of Pruning
`python Pfinetune.py --epoch 300`
##  Finetune of Quantization 
`python Qfinetune.py --epoch 100`  
##  Test
`python test1.py`
