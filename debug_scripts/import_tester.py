import os
import sys
import importlib.util
from pathlib import Path
def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(package_name)
        else:
            importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Other error: {str(e)}"
def test_script_imports(script_path):
    """Test imports in a Python script by parsing the file"""
    import_issues = []
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        # Simple parsing to find import statements
        lines = content.split('\n')
        import_lines = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
        for import_line in import_lines:
            try:
                # Try to extract module name
                if import_line.startswith('from '):
                    parts = import_line.split()
                    if len(parts) >= 2:
                        module_name = parts[1]
                        success, error = test_import(module_name)
                        if not success:
                            import_issues.append({
                                'import_line': import_line,
                                'error': error
                            })
                elif import_line.startswith('import '):
                    parts = import_line.split()
                    if len(parts) >= 2:
                        module_name = parts[1].split('.')[0]  # Get base module
                        success, error = test_import(module_name)
                        if not success:
                            import_issues.append({
                                'import_line': import_line,
                                'error': error
                            })
            except Exception as e:
                import_issues.append({
                    'import_line': import_line,
                    'error': f"Parsing error: {str(e)}"
                })
        return import_issues
    except Exception as e:
        return [{'import_line': 'File reading', 'error': str(e)}]
def test_common_libraries():
    """Test common data science and ML libraries"""
    libraries = {
        # Core data science
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        # ML libraries
        "scikit-learn": "sklearn",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "imbalanced-learn": "imblearn",
        # Deep learning
        "torch": "torch",
        "torchvision": "torchvision",
        # Utilities
        "joblib": "joblib",
        "pickle": "pickle",
        "json": "json",
        "yaml": "yaml"
    }
    results = {}
    for name, module in libraries.items():
        success, error = test_import(name, module)
        results[name] = {
            "success": success,
            "error": error if not success else None
        }
    return results
def test_project_scripts():
    """Test all project Python scripts"""
    project_root = os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
    script_results = {}
    # Find all Python files in the project
    python_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip certain directories
        dirs_to_skip = ['__pycache__', '.git', 'logs']
        dirs[:] = [d for d in dirs if d not in dirs_to_skip]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    print(f"Found {len(python_files)} Python files to test\n")
    for script_path in python_files:
        relative_path = os.path.relpath(script_path, project_root)
        print(f"Testing imports in {relative_path}...")
        issues = test_script_imports(script_path)
        script_results[relative_path] = {
            "status": "success" if not issues else "issues",
            "issues": issues,
            "issue_count": len(issues)
        }
        if issues:
            print(f"  Found {len(issues)} import issues")
            for issue in issues:
                print(f"    {issue['import_line']}: {issue['error']}")
        else:
            print("  All imports successful")
        print()
    return script_results
def main():
    print("=" * 60)
    print("PYTHON IMPORT VALIDATION FOR REALTIME CREDIT CARD FRAUD DETECTION")
    print("=" * 60)
    # Test common libraries
    print("Testing common data science and ML libraries...")
    library_results = test_common_libraries()
    print("\nLibrary Test Results:")
    print("-" * 30)
    missing_libraries = []
    for library, result in library_results.items():
        if result["success"]:
            print(f"[OK] {library}: Available")
        else:
            print(f"[FAIL] {library}: MISSING ({result['error']})")
            missing_libraries.append(library)
    if missing_libraries:
        print(f"\nMissing libraries: {', '.join(missing_libraries)}")
    else:
        print("\nAll common libraries are available!")
    print("\n" + "=" * 60)
    print("Testing Project Script Imports")
    print("=" * 60)
    # Test project scripts
    script_results = test_project_scripts()
    print("=" * 60)
    print("IMPORT VALIDATION SUMMARY")
    print("=" * 60)
    # Count issues
    total_scripts = len(script_results)
    scripts_with_issues = sum(1 for r in script_results.values() if r["issue_count"] > 0)
    print(f"Total Python scripts tested: {total_scripts}")
    print(f"Scripts with import issues: {scripts_with_issues}")
    if scripts_with_issues > 0:
        print("\nScripts with issues:")
        for script, result in script_results.items():
            if result["issue_count"] > 0:
                print(f"  {script}: {result['issue_count']} issues")
    else:
        print("\nAll project scripts import successfully!")
    # Save results
    results = {
        "library_results": library_results,
        "script_results": script_results
    }
    import json
    with open('./reports/import_validation_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to ./reports/import_validation_report.json")
if __name__ == "__main__":
    main()
