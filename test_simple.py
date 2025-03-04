"""
Simple test script for the project structure.
This doesn't require any dependencies.
"""

import os
import sys

def check_project_structure():
    """Check if all the expected files exist."""
    expected_files = [
        '.env',
        'app.py',
        'data_cache.py',
        'irrigation.py',
        'openet_api.py',
        'requirements.txt',
        'utils.py',
        'README.md',
        'tests/test_irrigation.py',
        'tests/test_openet_api.py'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Error: The following files are missing: {', '.join(missing_files)}")
        return False
    
    print("All expected files are present.")
    return True

def check_file_content():
    """Check if the files have content."""
    files_to_check = [
        'app.py',
        'data_cache.py',
        'irrigation.py',
        'openet_api.py',
        'utils.py'
    ]
    
    empty_files = []
    for file_path in files_to_check:
        if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
            empty_files.append(file_path)
    
    if empty_files:
        print(f"Error: The following files are empty: {', '.join(empty_files)}")
        return False
    
    print("All files have content.")
    return True

def check_env_file():
    """Check if the .env file has the API key."""
    if not os.path.exists('.env'):
        print("Error: .env file is missing")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
    
    if 'OPENET_API_KEY=' not in content:
        print("Error: .env file does not contain the API key")
        return False
    
    print(".env file is correctly set up with the API key.")
    return True

def main():
    """Run all checks."""
    print("Testing project structure...")
    
    all_passed = True
    all_passed &= check_project_structure()
    all_passed &= check_file_content()
    all_passed &= check_env_file()
    
    if all_passed:
        print("\nAll tests passed! The project structure is correct.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()