from setuptools import setup


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='cco',
   version='0.2',
   description='Chance-constrained optimization',
   author='Anonymous',
   author_email='Anonymous@Anonymous',
   packages=['cco'],
   url="https://github.com/AnnonymousAuthor/KDD398",
   install_requires=['numpy', 'pandas', 'scipy'], #external dependent packages 
)
