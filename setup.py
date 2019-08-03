from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='CSDGAN',
    version='1.0.0',
    author='Andrew Gray',
    author_email='aj.gray619@gmail.com',
    description='Conditional Synthetic Data Generative Adversarial Network (CSDGAN)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Atrus619/CSDGAN',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask',
        'flask_moment',
        'pandas',
        'torch',
        'torchvision',
        'scikit-learn',
        'numpy',
        'redis',
        'rq',
        'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
