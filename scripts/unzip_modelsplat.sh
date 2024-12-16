#!/bin/bash
# Directory containing the .zip files
SOURCE_DIR="../gs_data/modelsplat/"
# Directory where all extracted files will be merged
DEST_DIR="../gs_data/modelsplat/modelsplat_ply"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# check if SOURCE_DIR is exist
if [ ! -d "$SOURCE_DIR" ]; then
    echo "$SOURCE_DIR does not exist."
    exit 1
fi


for file in "$SOURCE_DIR"/*.zip; 
do
  # Extract the .zip file
  base_name=$(basename "$file" .zip)
  target_dir="$DEST_DIR/$base_name"

  unzip "$file" -d "$target_dir"

done

