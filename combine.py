import os

# Define the root directory
root_dir = "./"  # Replace this with your root directory
output_file = "merged_files.txt"

# Define folders and file types to exclude
exclude_dirs = {"venv", "data", ".git", "logs"}
exclude_extensions = {".ipynb", ".csv", ".log"}

# Open the output file to write the merged content
with open(output_file, "w", encoding="utf-8") as outfile:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude specified directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            # Skip files with excluded extensions
            if any(filename.endswith(ext) for ext in exclude_extensions):
                continue
            
            # Construct the full path of the file
            file_path = os.path.join(dirpath, filename)
            # Get the relative path from the root
            relative_path = os.path.relpath(file_path, start=root_dir)
            
            # Write the relative path and separator
            outfile.write(f"./{relative_path} ===================\n")
            
            # Read and write the content of the file
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"Error reading file: {e}\n")
            
            # Add a newline separator for readability
            outfile.write("\n\n")

print(f"Files merged into {output_file}")
