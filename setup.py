from setuptools import setup

setup(
    name='rltime',
    version='0.1',
    install_requires=[
        'gym',
        'cloudpickle',
        'opencv-python',
    ],
    packages=["rltime"],
)
