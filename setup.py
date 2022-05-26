from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy==1.16.5', 'keras==2.3.1', 'scikit-image==0.15.0', 'tensorflow==2.7.2', 'google-cloud-storage==1.23.0', 'opencv-python==4.1.2.30']

setup(
    name='unet',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='unet model'
)