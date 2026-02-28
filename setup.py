# This folder makes this project behave like a professional Python package.

from setuptools import find_packages, setup

HYPEN_EDOT='-e .'
def getRequirements(filePath):
    requirements=[]
    with open(filePath) as file:
        requirements=file.readlines()
        requirements=[req.replace('\n', '') for req in requirements]
        
        if HYPEN_EDOT in requirements:
            requirements.remove(HYPEN_EDOT)
    
    return requirements

setup(
    name='MalariaDetection',
    version='0.0.1',
    author='Group',
    packages=find_packages(),
    install_requires=getRequirements('requirements.txt')
)