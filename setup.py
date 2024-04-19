from setuptools import setup, find_packages

setup(
    name='BreastCancerMLPModel',  
    version='0.0.30',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'scikit-learn==1.4.2',
        'numpy==1.25.2'
    ],
    author='Sergio Sánchez Sánchez',
    author_email='dreamsoftware92@gmail.com',
    description='A package for breast cancer diagnosis using MLP classifier.',
    url='https://github.com/sergio11/breast_cancer_diagnosis_mlp',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7, <4',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)

"""
BreastCancerMLPModel Setup

This setup script configures the installation of the BreastCancerMLPModel package. BreastCancerMLPModel is a package for breast cancer diagnosis using MLP classifier.

Project Details:
- Name: BreastCancerMLPModel
- Version: 0.0.30
- Author: Sergio Sánchez Sánchez
- Email: dreamsoftware92@gmail.com
- Description: A package for breast cancer diagnosis using MLP classifier.
- Repository: https://github.com/sergio11/breast_cancer_diagnosis_mlp

Development Status: Beta

License: MIT License

Python Version Compatibility: >=3.7, <4

For more details, please refer to the project's README.md file.
"""