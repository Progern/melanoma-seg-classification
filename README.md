# Melanoma Classification using a Cascade of Convolutional Neural Networks 🪀

## General information

This repository contains the code to train, compare and evaluate two types of models:
- Segmentation models: yield masks that denote the birthmarks on human skin
- Classification models: yield labels whether the input image contains simple birthmark or a melanoma cancer mark


The models were trained separately and then combiend in a cascaded pipeline. The rationale for using two models is the next:
- The first model provides binary segmentation of the birthmark and yields a mask that denotes the mark. Applying the mask to the input image allows us to propagate only useful signal further in the system.
- The classification model was trained on masked images, so it expects such an image as an input. Using masked images makes it easier for the model to provide accurate results, as it concentrates only on the birthmark and not on the skin around the mark.

## Segmentation 🔬

### Training🏋️

To train a model you would need to navigate your terminal to ROOT_DIR/scripts and run 

```bash
python seg_train_pipeline.py
```
There is a number of input arguments that are optional, and if not provided the system will use the default ones:

- Number of epochs (default: 20) 
- Device - cuda or cpu (default: cuda)
- Encoder backbone for the network (default: xception, other options: [any from SMP Zoo](https://github.com/qubvel/segmentation_models#models-and-backbones))
- Optimizer (default: sgd, other options: adam) 
- Learning rate (default: 0.0001) 
- Momentum (default: 0.99) 
- Weight decay (default: 0.9) 

An example of running training with non-default parameters

```bash
python seg_train_pipeline.py --epochs 20 --encoder mobilenet --optimizer adam --learning_rate 0.001 --momentum 0.9 --weight_decay 0.5
```

### Inference 🧘‍

This system uses mean Intersection-over-Union (mIoU) as an evaluation metric. To run inference and obtain test-time metrics one needs to provide model path and data path:
```bash
python seg_inference_pipeline.py --path ../models/path_to_model_checkpoint.pth --data_path ../dataset/
```

## Classification 🐈🐕

### Training 🏋️‍♀️

To train a model you would need to navigate your terminal to ROOT_DIR/scripts and run 

```bash
python classification_train_pipeline.py
```
There is a number of input arguments that are optional, and if not provided the system will use the default ones:

- Number of epochs (default: 20) 
- Device - cuda or cpu (default: cuda)
- Pre-trained ImageNet architecture (default: inception, other options: inception, resnet, googlenet)
- Class weight, for imbalanced classification problems (default: 1, no weighting). The system automatically founds the undersampled class and gives it a bigger weight in the Cross Entropy Loss.

An example of running training with non-default parameters

```bash
python classification_train_pipeline.py --epochs 20 --architecture resnet --class_weight 3
```

### Inference 🧘‍♀️

This system uses accuracy, precision, recall, confusion matrix and F1 score as evaluation metrics. To run inference and obtain test-time metrics one needs to navigate to ROOT_DIR/notebooks and follow the _classification_inference_pipeline.ipynb_.
