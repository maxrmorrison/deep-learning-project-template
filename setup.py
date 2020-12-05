from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='NAME',
    description='',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/NAME',
    install_requires=['pytorch-lightning'],
    packages=['NAME'],
    package_data={'NAME': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
