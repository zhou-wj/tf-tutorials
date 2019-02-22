## Introduce
- a project for SVHN(Street View House Numbers) classification  

## Data examples
[!train-data](images/train_data.png)
[!test-data](images/test_data.png)

## Training commands
- training from scratch: python train.py
- training from checkpoint: python train.py -c train\_log/models

## Tensorboard commands
> tensorboard --logdir ./trai\_log/tf\_log/
