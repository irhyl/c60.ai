import os

def create_project_structure(root_dir="c60ai"):
    """
    Create the folder structure for the C60.ai AutoML project.
    
    Args:
        root_dir (str): Root directory name for the project. Defaults to 'c60ai'.
    """
    # Define the folder structure
    directories = [
        f"{root_dir}/docs",
        f"{root_dir}/engine",
        f"{root_dir}/interface",
        f"{root_dir}/memory/graphs",
        f"{root_dir}/notebooks",
        f"{root_dir}/tests",
        f"{root_dir}/datasets",
        f"{root_dir}/deploy/c60toolkit",
        f"{root_dir}/config",
    ]
    
    # Create each directory
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print(f"\nFolder structure for {root_dir} created successfully!")

if __name__ == "__main__":
    create_project_structure()