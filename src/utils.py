import os
import glob
import random
import json
from datetime import datetime
import nibabel as nib
import matplotlib.pyplot as plt
import torch
def create_data_split_json(image_path, label_path, split_ratio, data_split_path, seed = 42):
    data_dicts = get_data_dict(image_path, label_path)
    split_data = data_split(data_dicts, split_ratio=split_ratio, seed = seed)
    if not os.path.exists(data_split_path):
        with open(data_split_path, 'w') as f:
            json.dump(split_data, f)
        print(f'file saved to {data_split_path}')
    else:
        print(f'file already exists at {data_split_path}')
    return split_data

def get_data_dict(image_path, label_path):
    """
    Given the paths to the directories containing images and labels, this function
    returns a list of dictionaries where each dictionary contains the paths to an image
    and its corresponding label.

    Args:
        image_path (str): Path to the directory containing image files.
        label_path (str): Path to the directory containing label files.

    Returns:
        list: A list of dictionaries, each containing 'image' and 'label' keys with corresponding file paths.
    """
    if label_path is not None:
        print("===== Loading both image and label data =====")
        # Verify that the provided paths exist
        assert os.path.exists(image_path), f"Image path {image_path} does not exist."
        assert os.path.exists(label_path), f"Label path {label_path} does not exist."
        # Get sorted lists of image and mask file paths
        image_path_list = sorted(glob.glob(os.path.join(image_path, '*.nii.gz')))
        mask_path_list = sorted(glob.glob(os.path.join(label_path, '*.nii.gz')))
        # Ensure the number of images and masks are the same
        assert len(image_path_list) == len(mask_path_list), "The number of images and masks must be the same"
        # Create a list of dictionaries with image and label paths
        data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(image_path_list, mask_path_list)]
        return data_dicts
    else: 
        print('===== Loading only image data =====')
        assert os.path.exists(image_path), f"Image path {image_path} does not exist."
        image_path_list = sorted(glob.glob(os.path.join(image_path, '*.nii.gz')))
        data_dicts = [{"image": image_name} for image_name in image_path_list]
        return data_dicts



def data_split(data_dicts, split_ratio=[0.8, 0.1, 0.1], seed=42):
    """
    Splits the data into training, validation, and test sets based on the provided split ratios.

    Args:
        data_dicts (list): List of dictionaries containing 'image' and 'label' keys.
        split_ratio (list): List containing the ratios for splitting the data. 
                            Should sum to 1.0. Default is [0.8, 0.1, 0.1].
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: Three lists containing the training, validation, and test data dictionaries.
    """
    # Ensure the split ratios sum to 1
    assert sum(split_ratio) == 1.0, "The sum of split_ratio should be 1"
    
    # Set the random seed for reproducibility
    random.seed(seed)
    # Shuffle the data_dicts to ensure randomness
    random.shuffle(data_dicts)
    
    # Get the total size of the data
    data_size = len(data_dicts)
    
    # Calculate the size of the training set
    train_size = int(split_ratio[0] * data_size)
    
    # Split the data based on the number of provided ratios
    if len(split_ratio) == 2:
        # If only two ratios are provided, split into training and validation sets
        train_data = data_dicts[:train_size]
        val_data = data_dicts[train_size:]
        test_data = []
    elif len(split_ratio) == 3:
        # If three ratios are provided, split into training, validation, and test sets
        val_size = int(split_ratio[1] * data_size)
        train_data = data_dicts[:train_size]
        val_data = data_dicts[train_size:val_size + train_size]
        test_data = data_dicts[val_size + train_size:]
    else:
        # Raise an error if the number of ratios is not 2 or 3
        raise ValueError("split_ratio must have 2 or 3 items")
    
    # Shuffle the data to ensure randomness
    random.shuffle(data_dicts)
    
    # Print the sizes of the splits
    print("Train data size: ", len(train_data))
    print("Validation data size: ", len(val_data))
    print("Test data size: ", len(test_data))
    split_data = {'train': train_data, 'val': val_data, 'test': test_data}
    return split_data


def load_split_data_dicts(data_split_path):
    """
    Loads the data split from a JSON file and verifies that there is no overlap 
    between the training, validation, and test sets.

    Args:
        data_split_path (str): Path to the JSON file containing the data split.

    Returns:
        dict: A dictionary containing the training, validation, and test data dictionaries.
    """
    # Verify that the provided path exists
    assert os.path.exists(data_split_path), "data_split.json does not exist"
    
    # Load the data split from the JSON file
    with open(data_split_path, 'r') as f:
        data_split = json.load(f)
    
    # Extract the training, validation, and test data
    train_data = data_split.get('train')
    val_data = data_split.get('val')
    test_data = data_split.get('test')
    
    # Extract image paths from the data splits
    train_image = [data['image'] for data in train_data] if train_data!=None else []
    val_image = [data['image'] for data in val_data] if val_data!=None else []
    test_image = [data['image'] for data in test_data] if test_data!=None else []
    
    # Check for overlap between the training, validation, and test sets
    assert len(set(train_image).intersection(set(val_image))) == 0, "Train and val data should not overlap"
    assert len(set(train_image).intersection(set(test_image))) == 0, "Train and test data should not overlap"
    assert len(set(val_image).intersection(set(test_image))) == 0, "Val and test data should not overlap"
    
    # Print confirmation that the data split is correct
    print("===== Loading data split path: ", data_split_path)
    print("Data split is correct and no overlap between train, val and test data")
    print("Train data size: ", len(train_data)) if train_data!=None else print("Train data size: 0")
    print("Validation data size: ", len(val_data)) if val_data!=None else print("Validation data size: 0")
    print("Test data size: ", len(test_data)) if test_data!=None else print("Test data size: 0")
    
    return data_split


def get_current_time():
    """
    Returns the current time in the format "YYYY-MM-DD_HH-MM-SS".

    Returns:
        str: The current time as a formatted string.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def initialize_run(cfg, dm):
    # Initialize run name
    version = get_current_time() # Initialize version based on current time
    run_name = f"{cfg.model.model_name}_{cfg.dataset.task_name}_tr{dm.num_train}_val{dm.num_val}_ts{dm.num_test}_{version}_{cfg.logger.tag}"
    project_name = cfg.logger.project_name # a project name in W&B (related to research project carried out)
    # Create base directory using the run name
    base_dir = os.path.join(cfg.logger.log_dir, 
                            project_name, 
                            run_name
                            )
    os.makedirs(base_dir, exist_ok=True)
    # Create subdirectories for W&B logs and checkpoints
    wandb_log_dir = os.path.join(base_dir, "wandb_logs")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(wandb_log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    return run_name, version, project_name, wandb_log_dir, checkpoints_dir

def ensure_folder_exists(folder_path):
    """
    Check if the input folder exists, if not, create the folder.
    Parameters:
    folder_path (str): Path to the folder to check or create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Folder created: {folder_path}')
    else:
        print(f'Folder already exists: {folder_path}')

def save_nii_nib(dir, data, affine, name):
    '''
    dir: output path
    data: need to be numpy array
    affine: affine information of input data
    name: file name
    '''
    assert os.path.exists(dir)
    data_nib = nib.Nifti1Image(data, affine)
    # assert data_nib.get_data_dtype() == np.dtype(np.int16) or np.dtype(np.float32)
    nib.save(data_nib, os.path.join(dir,name))
    print(f'file {name} saved')
    
    

def visualize_3d_tensor_during_training(image, label, output, slice_idx=50):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].set_title("image")
    axes[0].imshow(image[:, :, slice_idx], cmap="gray")
    axes[1].set_title("label")
    axes[1].imshow(label[:, :, slice_idx])
    axes[2].set_title("output")
    axes[2].imshow(output[:,:,slice_idx])
    plt.show()
    
def visualize_3d_tensor_during_inferencing(image, output, slice_idx=50):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].set_title("image")
    axes[0].imshow(image[:, :, slice_idx], cmap="gray")
    axes[1].set_title("output")
    axes[1].imshow(output[:,:,slice_idx])
    plt.show()