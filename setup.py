# encoding: utf-8
from setuptools import setup, find_packages

setup(name='lyz',
      version='0.2',
      description='Analysis methods for neural representations',
      url='https://github.com/gchrupala/lyz',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      zip_safe=False,
      packages=find_packages(exclude='test'),
      install_requires=[
          'torch>=1.2.0',
          'torchvision>=0.4.0',
          'numpy>=1.17.2',
          'scikit-learn',
          'plotnine',
          'ursa'
      ]
     )
