from setuptools import setup

setup(
    name='rltime',
    version='0.1',
    author="Opher Lieber",
    author_email="opherlie@gmail.com",
    install_requires=[
        'gym',
        'cloudpickle',
        'opencv-python',
    ],
    packages=["rltime"],
    python_requires='>=3.6'
)
