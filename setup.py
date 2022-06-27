from setuptools import setup

# https://pythonhosted.org/an_example_pypi_project/setuptools.html

setup(
 name='dl_experiments_tf',
 version='1.0.0',
 author='EC',
 author_email='github@procurasia.com',
 packages=['dl_experiments_tf'],
 scripts=[],
 url='',
 license='LICENSE',
 description='Set of utility functions and tensorflow functions to experiment with DL',
 long_description=open('README.md').read(),
 classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "Programming Language :: Python :: 3.8",
    "Topic :: Utilities",
],
#  install_requires=[
#    "bokeh",
#    "IPython",
#    "matplotlib",
#    "numpy",
#    "pandas",
#    "scipy"
#  ],
)
