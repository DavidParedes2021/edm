import os
import shutil

def copy_first_n_files(src_folder, dst_folder, n):
    # Ensure destination exists
    os.makedirs(dst_folder, exist_ok=True)
    
    # Get only files (exclude directories) and sort for consistency
    files = sorted([
        f for f in os.listdir(src_folder)
        if os.path.isfile(os.path.join(src_folder, f))
    ])
    
    # Take first n files
    selected_files = files[:n]
    
    # Copy files
    for file_name in selected_files:
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)
        shutil.copy2(src_path, dst_path)  # preserves metadata
    
    print(f"Copied {len(selected_files)} files to {dst_folder}")


# Example usage
copy_first_n_files( r"../../data/datasets/kvasir_classified/none",  r"../../data/datasets/kvasir_classified/none_filtered", 3791)