from setuptools import setup, find_packages

setup(
    name='wirehead',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pymongo',
        'torch',
        'numpy',
        'PyYaml',
        # List your package dependencies here
        # Example: 'numpy>=1.19.0',
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line entry points here
            # Example: 'your-command = your_package.module:function',
        ],
    },
)
