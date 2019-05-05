from setuptools import setup, find_packages

readme = open('README.md').read()
setup(name='gepapy',
      version='0.3.22',
      description='Good-enough polynomial approximation.',
      long_description=readme,
      url='https://github.com/nishbo/gepa',
      author='Anton Sobinov',
      author_email='an.sobinov@gmail.com',
      license='Apache 2.0',
      packages=find_packages())
