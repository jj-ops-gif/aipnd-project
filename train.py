#!/usr/bin/env python3
#
#   Example call:
#    python train.py data_dir --save_dir save_directory --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu
##

# Imports argparse python module
import argparse
from model_util import *
from file_util import *

# Main program function defined below
def main():
    in_arg = get_input_args()

    # data_dir = '/home/ubuntu/.cache/kagglehub/datasets/flowers'
    train_folder = Path(in_arg.data_dir + '/train')
    valid_folder = Path(in_arg.data_dir + '/valid')
    if not train_folder.is_dir() or not train_folder.exists() or not valid_folder.is_dir() or not valid_folder.exists():
        print("Data directory does not exist.")
        return

    checkpoint_folder = Path(in_arg.save_dir)
    if not checkpoint_folder.is_dir() or not checkpoint_folder.exists():
        print("Checkpoint directory does not exist.")
        return
    
    train_datasets, train_loaders = load_train_data(str(train_folder))
    valid_datasets, valid_loaders = load_valid_data(str(valid_folder))

    num_classes = len(train_datasets.classes)
    model = update_model_with_hidden_layer(in_arg.arch, num_classes, in_arg.hidden_units)
    
    optimizer_state_dict, train_loss, valid_accuracy = train_model(model, train_loaders, valid_loaders, in_arg.epochs)

    checkpoint_filepath = Path(str(checkpoint_folder) + '/' + generate_filename() + '.pth')
    model_state_dict = model.state_dict()
    class_to_idx = train_datasets.class_to_idx
    checkpoint = save_checkpoint(str(checkpoint_filepath), model_state_dict, 
                                 in_arg.arch, num_classes, in_arg.hidden_units,
                                 optimizer_state_dict, class_to_idx, train_loss, valid_accuracy, in_arg.epochs)

def get_input_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 0: Data directory
    parser.add_argument("data_dir", type=str, help="Path to the data folder where you can find /train and /valid sub folders (mandatory)")
    # Argument 1: Path to the folder to save checkpoint
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Path to the folder to save checkpoint') 
    # Argument 2: Pre-trained model architecture
    parser.add_argument('--arch', type=str, default='vgg16', help='Pre-trained model architecture') 
    # Argument 3: Learning rate - float
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate - float') 
    # Argument 4: Hidden unit - int
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden unit - int')
    # Argument 5: Epochs - int
    parser.add_argument('--epochs', type=int, default=20, help='Epochs - int')
    # Argument 6: GPU - int
    parser.add_argument('--gpu', action="store_true", help='GPU mode (default: False)')
    
    # Assigns variable in_args to parse_args()
    args = parser.parse_args()

    # Print parameter values
    print("data_dir: ", args.data_dir)
    print("save_dir: ", args.save_dir)
    print("arch: ", args.arch)
    print("learning_rate: ", args.learning_rate)
    print("hidden_units: ", args.hidden_units)
    print("epochs: ", args.epochs)
    print("gpu: ", args.gpu)

    return args

# Call to main function to run the program
if __name__ == "__main__":
    main()