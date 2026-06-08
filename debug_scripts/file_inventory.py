import os
import json
def categorize_files():
    project_root = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Initialize categories
    file_inventory = {
        "python_scripts": {
            "data_processing": [],
            "model_training": [],
            "evaluation": [],
            "utilities": [],
            "other": []
        },
        "data_files": {
            "raw": [],
            "processed": [],
            "enhanced": []
        },
        "model_artifacts": [],
        "reports": [],
        "config_docs": [],
        "other": []
    }
    # Walk through all files
    for root, dirs, files in os.walk(project_root):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, project_root)
            # Categorize by extension and location
            if file.endswith('.py'):
                if 'data/src' in file_path or 'src' in file_path:
                    file_inventory["python_scripts"]["data_processing"].append(relative_path)
                elif 'model/src' in file_path:
                    file_inventory["python_scripts"]["model_training"].append(relative_path)
                elif 'final_model_evaluation' in file_path:
                    file_inventory["python_scripts"]["evaluation"].append(relative_path)
                elif 'utils' in file_path:
                    file_inventory["python_scripts"]["utilities"].append(relative_path)
                else:
                    file_inventory["python_scripts"]["other"].append(relative_path)
            elif file.endswith('.csv'):
                if 'raw' in file_path:
                    file_inventory["data_files"]["raw"].append(relative_path)
                elif 'enhanced' in file_path:
                    file_inventory["data_files"]["enhanced"].append(relative_path)
                elif 'processed' in file_path:
                    file_inventory["data_files"]["processed"].append(relative_path)
            elif file.endswith(('.pkl', '.txt')):
                if 'models' in file_path:
                    file_inventory["model_artifacts"].append(relative_path)
                else:
                    file_inventory["other"].append(relative_path)
            elif file.endswith(('.json', '.html')):
                if 'reports' in file_path or 'json' in file_path:
                    file_inventory["reports"].append(relative_path)
                else:
                    file_inventory["config_docs"].append(relative_path)
            elif file.endswith(('.png', '.md')):
                if 'reports' in file_path:
                    file_inventory["reports"].append(relative_path)
                else:
                    file_inventory["other"].append(relative_path)
    return file_inventory
def print_inventory():
    inventory = categorize_files()
    project_root = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"=== FILE INVENTORY FOR {project_root} ===\n")
    print("PYTHON SCRIPTS:")
    print("  Data Processing:")
    for file in inventory["python_scripts"]["data_processing"]:
        print(f"    - {file}")
    print("  Model Training:")
    for file in inventory["python_scripts"]["model_training"]:
        print(f"    - {file}")
    print("  Evaluation:")
    for file in inventory["python_scripts"]["evaluation"]:
        print(f"    - {file}")
    print("  Utilities:")
    for file in inventory["python_scripts"]["utilities"]:
        print(f"    - {file}")
    print("  Other:")
    for file in inventory["python_scripts"]["other"]:
        print(f"    - {file}")
    print("\nDATA FILES:")
    print("  Raw:")
    for file in inventory["data_files"]["raw"]:
        print(f"    - {file}")
    print("  Processed:")
    for file in inventory["data_files"]["processed"]:
        print(f"    - {file}")
    print("  Enhanced:")
    for file in inventory["data_files"]["enhanced"]:
        print(f"    - {file}")
    print("\nMODEL ARTIFACTS:")
    for file in inventory["model_artifacts"]:
        print(f"    - {file}")
    print("\nREPORTS AND DOCUMENTATION:")
    for file in inventory["reports"]:
        print(f"    - {file}")
    print("\nCONFIGURATION AND DOCUMENTATION:")
    for file in inventory["config_docs"]:
        print(f"    - {file}")
    print("\nOTHER FILES:")
    for file in inventory["other"]:
        print(f"    - {file}")
if __name__ == "__main__":
    print_inventory()
