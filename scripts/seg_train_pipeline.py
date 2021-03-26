import torch
import constants as const

from argparse import ArgumentParser
from seg_train_utils import *


def main(epochs, 
         encoder, 
         optimizer, 
         learning_rate, 
         momentum,
         weight_decay):
    
    print("Loading data...")
    train_dataset, val_dataset = load_data("../dataset")
    
    print("Creating model...")
    model_name = '../models/{}.pth'.format(generate_model_name(encoder = encoder))
    model = create_model(encoder = encoder)
    print(model_name)
    
    loss = get_loss()
    metrics = get_metrics()
    optimizer = get_optimizator(model, optimizer, 
                                learning_rate, momentum, weight_decay)
    print(optimizer)
    
    print("Establishing data pipelines...")
    train_loader = get_data_loader(train_dataset, 
                                   batch_size = const.batch_size_train, 
                                   num_workers = const.num_workers_train)
    
    valid_loader = get_data_loader(val_dataset, 
                                   batch_size = const.batch_size_val, 
                                   shuffle=False, 
                                   num_workers = const.num_workers_val)
    
    train_epoch = get_train_epoch(model, loss, metrics, optimizer, const.DEVICE)
    valid_epoch = get_valid_epoch(model, loss, metrics, const.DEVICE)
    
    max_score = 0
    
    # Callbacks
    es = EarlyStopping(patience=5)

    print("Training the model...")
    for i in range(0, epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        metric = valid_logs['iou_score']

        if max_score < metric:
            max_score = metric
            torch.save(model, model_name)
            print('Model saved!')

        if i % 5 == 0 and i != 0:
            optimizer.param_groups[0]['lr'] *= 1e-1
            print('Decrease decoder learning rate by 0.1!')
            
        if es.step(metric):
            break  # Early stopping
    
    print("Model trained succesfully.")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int,
                    help="number of epochs for training")
    
    parser.add_argument("-en", "--encoder", type=str,
                    help="pre-trained ImageNet encoder for the model")
    
    parser.add_argument("-opt", "--optimizer", type=str,
                    help="optimizer type. Possible values: sgd, adam.")
    
    parser.add_argument("-lr", "--learning_rate", type=float,
                    help="learning rate for the optimizer")
    
    parser.add_argument("-mom", "--momentum", type=float,
                    help="momentum value for the optimizer")
    
    parser.add_argument("-wd", "--weight_decay", type=float,
                    help="weight decay value for the optimizer")
    
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else const.epochs_default
    encoder = args.encoder if args.encoder is not None else const.ENCODER
    optimizer = args.optimizer if args.optimizer is not None else const.OPTIMIZER
    learning_rate = args.learning_rate if args.learning_rate is not None else const.LR
    momentum = args.momentum if args.momentum is not None else const.MOMENTUM
    weight_decay = args.weight_decay if args.weight_decay is not None else const.WD
    
    main(epochs, encoder, 
         optimizer, learning_rate, momentum, weight_decay)