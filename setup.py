from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'


def get_requirements(filename: str) -> List[str]:
    '''
    This function will return a list of requirements
    '''

    requirements = []

    with open(filename) as file:
        requirements=file.readlines()
        [requirement.replace('\n', '') for requirement in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup (
    name='MLPROJECT',
    version='0.0.1',
    author='Parv',
    author_email='pt21299@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
