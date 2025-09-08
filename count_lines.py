import glob
import os

def count_lines_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

def count_lines_in_directory(directory='.'):
    total_lines = 0
    python_files = glob.glob(os.path.join(directory, '**', '*.py'), recursive=True)
    
    for file in python_files:
        lines = count_lines_in_file(file)
        total_lines += lines
        
    return total_lines

if __name__ == '__main__':
    total = count_lines_in_directory()
    print(f'Total lines of Python code: {total}')