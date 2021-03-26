import os
import constants as const
from cls_train_utils import *


def main(epochs,
         device,
         architecture,
         class_weight,
         weighted: bool = True,
         scheduler: bool = True):
    print("Loading data...")
    train_dataset, train_data_split, val_dataset = load_data("../dataset")
    
    class_weights = get_default_class_weights(device)
    
    if weighted:
        print("Calculating data weights...")
        class_weights = get_class_weights(train_data_split, device, 
                                          underclass_scale_factor = class_weight)
        
    print("Class weights: {}".format(class_weights))
        
    print("Creating data loaders...")
    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)
    data_loaders_dict = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": train_dataset.__len__(), "val": val_dataset.__len__()}
    class_names = const.CL_CLASSES
    
    print("Defining metrics...")
    metric = F1(num_classes = len(const.CL_CLASSES))
    
    
    print("Creating model...")
    model_ft, criterion_ft, optimizer_ft, exp_lr_scheduler = get_model(architecture, 
                                                                       class_weights, 
                                                                       device)
    
    print("Starting training...")
    train_model(model_ft, data_loaders_dict, dataset_sizes, criterion_ft, 
                optimizer_ft, metric, exp_lr_scheduler, 
                architecture, device, epochs)
    
   
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int,
                    help="number of epochs for training")
    
    parser.add_argument("-a", "--architecture", type=str,
                    help="architecture to use as a backbone. Available: inception, resnet, squeezenet.")
    
    parser.add_argument("-d", "--device", type=str,
                    help="device to train with, either cpu or cuda:0")
    
    parser.add_argument("-cw", "--class_weight", type=int,
                    help="class weight multiplier for inbalances classes")
    
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else const.epochs_default
    architecture = args.architecture if args.architecture is not None else const.architecture_default
    device = args.device if args.device is not None else const.DEVICE
    class_weight = args.class_weight if args.class_weight is not None else 1
    
    main(epochs = epochs, 
         device = device,
         architecture = architecture,
         class_weight = class_weight)
