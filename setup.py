from setuptools import setup

all_info = {}
with open("EINSfit/version.py") as fp:
    exec(fp.read(), all_info)
    
__version__=all_info['__all__']['__version__']

setup(
    name='EINSfit',
    version=__version__,
    url='https://github.com/DominikZ/EINSfit-private/',
    author='Dominik Zeller',
    author_email='DominikZ@posteo.eu',
    license='GPLv3',
    description='EINSfit package',
    packages=['EINSfit',],    
    python_requires='>=3.6',
    install_requires=['lmfit >= 0.9.13', 'matplotlib >= 2', 'numpy >= 1.16', 'cycler>=0.10'],
)
