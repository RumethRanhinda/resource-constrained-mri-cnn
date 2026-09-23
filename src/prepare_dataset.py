import os
import random
from PIL import Image

def process_dataset(source_dir, dest_dir, target_size=(64, 64)):
    print(f"Processing dataset from {source_dir} to {dest_dir}...")
    
    # Classes and their mapping
    class_mapping = {
        'NonDemented': 'NonDemented',
        'VeryMildDemented': 'Demented',
        'MildDemented': 'Demented',
        'ModerateDemented': 'Demented'
    }
    
    # Collect all file paths mapped to their new classes
    class_files = {'NonDemented': [], 'Demented': []}
        
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        mapped_class = class_mapping.get(class_name)
        if not mapped_class:
            continue
            
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_files[mapped_class].append(os.path.join(class_path, file))
                
    # Prepare directories
    splits = ['train', 'val', 'test']
    for split in splits:
        for mapped_class in class_files.keys():
            os.makedirs(os.path.join(dest_dir, split, mapped_class), exist_ok=True)
            
    # Process each class
    for mapped_class, files in class_files.items():
        print(f"Found {len(files)} authentic images for {mapped_class} class.")
        
        # Shuffle for randomness in split
        random.seed(42) # Fixed seed for reproducibility
        random.shuffle(files)
        
        # Calculate split indices (70%, 15%, 15%)
        total = len(files)
        train_end = int(total * 0.7)
        val_end = train_end + int(total * 0.15)
        
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        split_dict = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        # Process and save images
        for split, split_files in split_dict.items():
            print(f"  Saving {len(split_files)} {mapped_class} images to {split}...")
            for i, src_file in enumerate(split_files):
                # Generate clean sequential filename
                # Example: Demented_train_0001.jpg
                filename = f"{mapped_class}_{split}_{i+1:04d}.jpg"
                dest_file = os.path.join(dest_dir, split, mapped_class, filename)
                
                # Resize and save
                try:
                    with Image.open(src_file) as img:
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        img_resized.save(dest_file)
                except Exception as e:
                    print(f"    Error processing {src_file}: {e}")

    print("Dataset preparation complete.")

if __name__ == "__main__":
    src = os.path.join(os.path.dirname(__file__), '..', 'data', 'OriginalDataset')
    dst = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    process_dataset(src, dst)
