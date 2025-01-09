#!/usr/bin/env python3
#
#   Example call:
#    python predict.py path_to_image checkpoint --top_k 3 --category_names cat_to_name.json --gpu
##

# Imports argparse python module
import argparse
# Imports utils
from model_util import *
from file_util import *

'''
Predict the class (or classes) of an image using a trained deep learning model.
'''
def main():
    in_arg = get_input_args()

    # Load checkpoint
    checkpoint = load_checkpoint(in_arg.checkpoint_path)
    class_to_idx = checkpoint['class_to_idx']         # Get class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}   # Reverse the mapping to create idx_to_class

    predict(in_arg.image_path, checkpoint, idx_to_class, in_arg.top_k)

'''
Get input arguments
'''
def get_input_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", type=str, help="Path to the input image (mandatory)")
    parser.add_argument("checkpoint_path", type=str, help="Checkpoint file path (mandatory)")
    parser.add_argument('--top_k', type=int, default=5, help='Top_k') 
    parser.add_argument('--category_names_path', type=str, default='cat_to_name.json', help='Path to the cat_to_name.json file')
    parser.add_argument('--gpu', action="store_true", help='GPU mode (default: False)')
    
    # Assigns variable in_args to parse_args()
    args = parser.parse_args()

    # Print parameter values
    print("image_path: ", args.image_path)
    print("checkpoint_path: ", args.checkpoint_path)
    print("top_k: ", args.top_k)
    print("category_names_path: ", args.category_names_path)
    print("gpu: ", args.gpu)

    return args


# Call to main function to run the program
if __name__ == "__main__":
    main()