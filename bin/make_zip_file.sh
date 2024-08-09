#!/bin/bash

# Check if the source directory is provided as a command-line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source_directory>"
    exit 1
fi

# Directory to be compressed
source_directory="$1"

# Zip file name
zip_file="stacking_framework.zip"

# Build folder
build_folder="tmp_build"

# Deleting the old build folder
rm -rf "$build_folder"

# Create the build folder
mkdir -p "$build_folder"

# Copy only *.py files to the build folder
find "$source_directory" -name "*.py" -exec cp --parents {} "$build_folder" \;

# Create the final zip file from the files in the build folder
zip -r "$zip_file" "$build_folder"

echo "Compression completed. Zip file: $zip_file"

echo "Content of the zip file: $(zipinfo $zip_file)"

# Removing the build folder
rm -rf "$build_folder"

