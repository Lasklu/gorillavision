from setuptools import find_packages, setup

setup(
   name='detector',
   version='0.1.0',
   author='An Awesome Coder',
   author_email='aac@example.com',
   package_dir = {"": "src"},
   packages = setuptools.find_packages(where="src"),
   url='http://pypi.python.org/pypi/PackageName/',
   description='An awesome package that does something',
   install_requires=[
       "seaborn",
   ],
)