import os
 
# Get the path of the directory
path = '/home/jm/Linux/workspace/YOLOv8/Lib_datasets/Stone_Classification/test/'
 
# List all files in the directory
files = os.listdir(path)
 
# Iterate over files in the directory
for file in files:
    print(file)
    # Check if the file is a directory
    if os.path.isdir(path+file):
        # Get the list of all files in a directory
        sub_files = os.listdir(path+file)
        print("sub_files")
        # Iterate over the files in the directory
        for sub_file in sub_files:
            # Print the file
            print(sub_file)
