import os

def check_and_remove_missing_files(input_file):
    # Read the text file containing the list of files to check
    all_lines = []
    files_to_check = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        
    for line in all_lines:
        # Split the line into path and the rest of the text (after the | character)
        parts = line.strip().split('|', 1)
        if parts:
            file_path = parts[0]
            files_to_check.append(file_path)
    
    # Check for missing files
    missing_files = []
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    # Print missing files
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(file)
        print(f"\nNumber of missing files: {len(missing_files)}")
        
        # Remove lines with missing files
        new_lines = []
        for line in all_lines:
            parts = line.strip().split('|', 1)
            if parts and parts[0] not in missing_files:
                new_lines.append(line)
        
        # Write the updated list back to the file
        with open(input_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"\nRemoved {len(missing_files)} entries from {input_file}")
    else:
        print("All files exist.")

# Provide the name of the text file containing the list
input_file = "combined_final.txt"  # Change to your actual filename
check_and_remove_missing_files(input_file)