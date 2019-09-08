from setuptools import setup, find_packages

setup(
    name='rltime',
    version='0.1.2',
    author="Opher Lieber",
    author_email="opherlie@gmail.com",
    install_requires=[
        'gym',
        'cloudpickle',
        'opencv-python',
    ],
    license="Apache 2.0",
    packages=[
        package for package in find_packages()
        if package.startswith('rltime')],
    include_package_data=True,
    package_data={'rltime': ['configs/*.json','configs/env_wrappers/*.json','configs/exploration/*.json','configs/models/*.json','configs/models/modules/*.json']},
    python_requires='>=3.6',
    description="RLtime is a reinforcement learning library focused on state-of-the-art q-learning algorithms and features",
    long_description="""
RLtime is a reinforcement learning library, currently supporting PyTorch, with focus on state-of-the-art q-learning algorithms and features, and interacting with real-time environments which require low-latency acting and sample-efficient training.
The latest code and instructions can be found here:
https://github.com/opherlieber/rltime
"""
)
