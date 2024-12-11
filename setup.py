from setuptools import setup, find_packages

setup(
   name='jaxvox',
   version='0.0.0',
   description='Voxel data structure for JAX',
   author='Charlie Gauthier',
   author_email='charlie-gauthier@outlook.com',
    packages=find_packages(),
   package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.pcd', '*.json'],
    },
    install_requires=["typing-extensions"]
)


