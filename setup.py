from setuptools import find_packages,setup
from typing import List


with open("README.MD", "r" , encoding="utf-8") as f:
    long_description = f.read()


def get_requirements()->List[str]:
    """
    Thiss function will return list of requirements
    
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

REPO_NAME = "kidney_disease_classification"
AUTHOR_USER_NAME = "pavani"

setup(
    name="cnnClassifier",
    version="0.0.1",
    author="Pavani kusuma",
    author_email="pavanikusuma2020@gmail.com",
    install_requires=get_requirements(),
    description = "A small python package for CNN app",
    long_description_content ="text/markdown",
    url =f"https://github.com/author/name",
    project_urls={"Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"},
    package_dir={"":"src"},
    packages= find_packages(where="src"),
)