import h5py
import os
import glob
import natsort 

def copy_h5_structure(source_group, destination_group):
    """
    Recursively copies the structure and datasets from a source group to a destination group,
    renaming 'action' to 'actions' and skipping 'num_step'.
    """
    for key, item in source_group.items():
        if isinstance(item, h5py.Group):
            # If the item is a group, create a new group in the destination
            # and recurse into it.
            new_group = destination_group.create_group(key)
            copy_h5_structure(item, new_group)
        elif isinstance(item, h5py.Dataset):
            # If the item is a dataset, check its name before copying.
            if key == 'num_step':
                # Skip the 'num_step' dataset.
                continue
            
            # Determine the new key name. If it's 'action', change it to 'actions'.
            new_key = 'actions' if key == 'action' else key
            
            # Copy the dataset to the destination group with the potentially new key.
            destination_group.create_dataset(new_key, data=item[...], dtype=item.dtype)

def merge_h5_files(source_folder, output_filepath):
    """
    Merges all HDF5 files from a source folder into a single HDF5 file.

    Each source file's content is placed into a 'demo_i' group under a 'data' group.

    Args:
        source_folder (str): The path to the folder containing the .h5 files.
        output_filepath (str): The path for the new, merged .h5 file.
    """
    # Find all .h5 files in the source folder.
    # The list is sorted numerically to ensure demo_0, demo_1, demo_2... order.
    source_files = natsort.natsorted(glob.glob(os.path.join(source_folder, '*.h5')))

    if not source_files:
        print(f"Error: No '.h5' files found in '{source_folder}'.")
        return

    print(f"Found {len(source_files)} files to merge.")

    try:
        # Create the new HDF5 file in write mode ('w').
        with h5py.File(output_filepath, 'w') as f_out:
            # Create the top-level 'data' group.
            data_group = f_out.create_group('data')
            print(f"Created output file: '{output_filepath}'")

            # Loop through each source file with an index.
            for i, file_path in enumerate(source_files):
                demo_name = f'demo_{i}'
                print(f"  -> Processing '{os.path.basename(file_path)}' into '/data/{demo_name}'")

                try:
                    # Open the source file in read mode ('r').
                    with h5py.File(file_path, 'r') as f_in:
                        # Create the destination group (e.g., 'data/demo_0').
                        demo_group = data_group.create_group(demo_name)
                        # Copy the entire structure and data from source to destination.
                        copy_h5_structure(f_in, demo_group)

                except Exception as e:
                    print(f"    [!] Warning: Could not process file {file_path}. Reason: {e}")
        
        print("\nMerge complete!")

    except Exception as e:
        print(f"An error occurred during the merge process: {e}")


def inspect_h5_file(file_path):
    """
    Inspects an HDF5 file and prints its complete structure including
    group names, dataset names, shapes, and data types.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"\n--- Verifying Structure of: {os.path.basename(file_path)} ---")
    try:
        with h5py.File(file_path, 'r') as f:
            print_hdf5_item(f, '')
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

def print_hdf5_item(item, prefix=''):
    """
    Recursively prints the name, shape, and type of items in an HDF5 Group.
    """
    # Limit the depth of inspection for large merged files
    if len(prefix) > 12: # Adjust this depth as needed (4 spaces * 3 levels)
        print(f"{prefix}[...]")
        return
        
    for key in sorted(item.keys()):
        if isinstance(item[key], h5py.Group):
            print(f"{prefix}[+] Group: {key}")
            print_hdf5_item(item[key], prefix + '    ')
        elif isinstance(item[key], h5py.Dataset):
            dataset = item[key]
            shape_str = f"Shape: {dataset.shape}"
            dtype_str = f"Dtype: {dataset.dtype}"
            print(f"{prefix}[-] Dataset: {key} ({shape_str}, {dtype_str})")
        else:
            print(f"{prefix}[?] Unknown item: {key}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. The folder where your 100 'demo_*.h5' files are located.
    SOURCE_DIRECTORY = '/home/carl_lab/diffusion_policy/pick_100'

    # 2. The full path and name for the new merged file you want to create.
    OUTPUT_FILE = 'pick_100.hdf5'
    
    # Run the merge process.
    merge_h5_files(SOURCE_DIRECTORY, OUTPUT_FILE)
    
    # Verify the structure of the newly created file.
    inspect_h5_file(OUTPUT_FILE)