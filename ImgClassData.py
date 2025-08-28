import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

class ImgClassData:
    """
    Data Parser for Image Classification
    """
    def __init__(self, filepath: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, debug = False):
        self.filepath = filepath
        
        # Get initial tree structure
        tree_json, leaf_files = self.folder_tree_json(filepath)
        img_dims = self.get_image_size(leaf_files, sample_rate=100)
        self.json_tree = json.dumps(tree_json, indent=2)
        self.IMSIZE = img_dims
        
        # Analyze dataset structure and splits
        dataset_splits = self.analyze_dataset_structure_and_splits(tree_json, filepath)
        self.DS_split_tree = json.dumps(dataset_splits, indent=2)
        
        # Check if we need to create splits automatically
        if not dataset_splits["train_test_val_split_exists"]:
            print("No existing splits detected. Creating train/val/test splits...")
            self.create_splits(train_ratio, val_ratio, test_ratio)
            # Re-analyze after creating splits
            tree_json, leaf_files = self.folder_tree_json(filepath)
            dataset_splits = self.analyze_dataset_structure_and_splits(tree_json, filepath)
            self.DS_split_tree = json.dumps(dataset_splits, indent=2)
        
        # Generate splits
        train_folders, val_folders, test_folders = self.gen_splits(dataset_splits)
        self.train_folders = train_folders
        self.val_folders = val_folders
        self.test_folders = test_folders
        
        # Set directory paths
        self.train_dir = os.path.dirname(train_folders[0]) if train_folders else None
        self.val_dir = os.path.dirname(val_folders[0]) if val_folders else None
        self.test_dir = os.path.dirname(test_folders[0]) if test_folders else None
        self.classes = [(os.path.basename(folder)) for folder in train_folders] if train_folders else []
        if debug:
            print(f"Image size: {self.IMSIZE}")
            print(f"Train folders: {self.train_folders}")
            print(f"Validation folders: {self.val_folders}")
            print(f"Test folders: {self.test_folders}")
            print(f"Classes: {self.classes}")
            
    def parse(self):
        # Implement parsing logic here
        pass

    def folder_tree_json(self, path):
        leaf_files = []
        path = Path(path)
        def build_tree(p):
            temp_leaf_files = []
            children = [c for c in p.iterdir() if c.is_dir()]
            is_leaf = len(children) == 0
            folder_count = len(children)
            if is_leaf:
                for f in p.iterdir():
                    if f.is_file():
                        temp_leaf_files.append(str(f))
                file_count = len(temp_leaf_files)
                leaf_files.extend(temp_leaf_files)
            else:
                file_count = None
            return {
                "folder_name": p.name,
                "is_leaf": is_leaf,
                "file_count": file_count,
                "folder_count": folder_count,
                "sub_folders": [build_tree(child) for child in children]
            }
        tree = build_tree(path)
        return tree, leaf_files
    
    def get_image_size(self, leaf_files, sample_rate=50):
        sampled_files = leaf_files[::sample_rate]
        sizes = []
        pxtodim = {}
        for file in sampled_files:
            try:
                with Image.open(file) as img:
                    pixels = 1
                    for dim in img.size:
                        pixels *= dim
                    pxtodim[pixels] = img.size
                    sizes.append(pixels)
            except Exception:
                continue

        if sizes:
            median_size = np.median(sizes)
            return pxtodim.get(median_size)
        return (224, 224)  # Default size if no images found
    
    def analyze_dataset_structure_and_splits(self, tree_json, filepath):
        """
        Analyze dataset structure to detect train/test/val splits and organize folder paths
        """
        # Check if dataset has train/test/val split structure
        split_keywords = ['train', 'test', 'val', 'validation']
        
        def has_split_structure(node):
            folder_names = [child['folder_name'].lower() for child in node.get('sub_folders', [])]
            return any(keyword in name for keyword in split_keywords for name in folder_names)
        
        has_splits = has_split_structure(tree_json)
        
        # Organize file paths
        dataset_splits = {
            "train_test_val_split_exists": has_splits,
            "splits": {}
        }
        
        if has_splits:
            # Organize by detected splits
            def organize_by_splits(node, current_path=""):
                for child in node.get('sub_folders', []):
                    child_name = child['folder_name'].lower()
                    path_key = f"{current_path}/{child['folder_name']}" if current_path else child['folder_name']
                    
                    # Determine split type
                    split_type = None
                    if 'train' in child_name:
                        split_type = 'train'
                    elif 'test' in child_name:
                        split_type = 'test'
                    elif 'val' in child_name or 'validation' in child_name:
                        split_type = 'val'
                    
                    if split_type and child.get('is_leaf'):
                        # Leaf folder with files
                        if split_type not in dataset_splits["splits"]:
                            dataset_splits["splits"][split_type] = {}
                        
                        class_name = child['folder_name']
                        if current_path:
                            folder_path = f"{filepath}/{current_path}/{child['folder_name']}"
                        else:
                            folder_path = f"{filepath}/{child['folder_name']}"
                        dataset_splits["splits"][split_type][class_name] = folder_path
                    
                    elif split_type and not child.get('is_leaf'):
                        # Split folder with class subfolders
                        if split_type not in dataset_splits["splits"]:
                            dataset_splits["splits"][split_type] = {}
                        
                        for class_folder in child.get('sub_folders', []):
                            if class_folder.get('is_leaf'):
                                class_name = class_folder['folder_name']
                                folder_path = f"{filepath}/{path_key}/{class_name}"
                                dataset_splits["splits"][split_type][class_name] = folder_path
                    
                    # Recursively check subfolders
                    organize_by_splits(child, path_key)
            
            organize_by_splits(tree_json)
        else:
            # No splits detected - put all files in training set
            dataset_splits["splits"]["train"] = {}
            
            def organize_all_as_train(node, current_path=""):
                if node.get('is_leaf'):
                    class_name = node['folder_name']
                    if current_path:
                        folder_path = f"{filepath}/{current_path}"
                    else:
                        folder_path = f"{filepath}"
                    dataset_splits["splits"]["train"][class_name] = folder_path
                
                for child in node.get('sub_folders', []):
                    child_path = f"{current_path}/{child['folder_name']}" if current_path else child['folder_name']
                    organize_all_as_train(child, child_path)
            
            organize_all_as_train(tree_json)
        
        return dataset_splits
    
    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Create train/val/test splits by moving files into new directory structure"""
        parent_folder = self.filepath
        # Ensure ratios sum to 1
        assert abs(float(train_ratio) + float(val_ratio) + float(test_ratio) - 1.0) < 1e-6

        # Create output folders
        for split in ["train", "val", "test"]:
            split_path = os.path.join(parent_folder, split)
            os.makedirs(split_path, exist_ok=True)

        # Loop through each class folder inside parent
        for class_name in os.listdir(parent_folder):
            class_path = os.path.join(parent_folder, class_name)
            
            # Skip the folders we just created
            if class_name in ["train", "val", "test"]:
                continue
            if not os.path.isdir(class_path):
                continue

            # Collect all files for this class
            files = os.listdir(class_path)
            random.shuffle(files)

            # Split indices
            n_total = len(files)
            n_train = int(train_ratio * n_total)
            n_val = int(val_ratio * n_total)

            train_files = files[:n_train]
            val_files = files[n_train:n_train + n_val]
            test_files = files[n_train + n_val:]

            # Function to move files into split folders
            def move_files(file_list, split):
                split_class_dir = os.path.join(parent_folder, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                for f in file_list:
                    src = os.path.join(class_path, f)
                    dst = os.path.join(split_class_dir, f)
                    shutil.move(src, dst)

            # Move files
            move_files(train_files, "train")
            move_files(val_files, "val")
            move_files(test_files, "test")
            
            # Remove empty original class folder
            try:
                os.rmdir(class_path)
            except OSError:
                pass  # Folder not empty, leave it
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test directories (legacy method for backwards compatibility)"""
        self.create_splits(train_ratio, val_ratio, test_ratio)
        
        train_dir = os.path.join(self.filepath, "train")
        val_dir = os.path.join(self.filepath, "val")
        test_dir = os.path.join(self.filepath, "test")

        return train_dir, val_dir, test_dir
    
    def gen_splits(self, dataset_splits):
        """Generate train/val/test folder lists from dataset splits"""
        data = dataset_splits
        train_folders = []
        val_folders = []
        test_folders = []
        
        # organize the data into splits
        has_splits = data["train_test_val_split_exists"]
        if has_splits:
            if "train" in data["splits"]:
                for value in data["splits"]["train"].values():
                    train_folders.append(value)
            if "val" in data["splits"]:
                for value in data["splits"]["val"].values():
                    val_folders.append(value)
            if "test" in data["splits"]:
                for value in data["splits"]["test"].values():
                    test_folders.append(value)
        else:
            # If no splits exist and auto_split is False, return empty lists
            # The user can manually call create_splits() or split_dataset() if needed
            pass

        return train_folders, val_folders, test_folders


if __name__ == "__main__":
    parser = ImgClassData("/Users/natejly/Desktop/PetImages", debug=True)
