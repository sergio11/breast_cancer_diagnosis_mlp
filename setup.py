from setuptools import setup, find_packages

setup(
    name='mlp-breast-cancer-diagnosis',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'scikit-learn==0.24.2',
        'numpy==1.21.2'
    ],
    author='DreamSoftware',
    author_email='dreamsoftware92@gmail.com',
    description='A package for breast cancer diagnosis using MLP classifier.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sergio11/breast_cancer_diagnosis_mlp',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
