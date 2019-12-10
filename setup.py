from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy==1.16.5', 'keras==2.3.1', 'scikit-image==0.14.5', 'tensorflow==2.0.0', 'google-cloud-storage==1.23.0']

setup(
    name='unet',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='unet model'
)