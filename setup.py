from setuptools import setup

setup(
    name='rltime',
    version='0.1.1',
    author="Opher Lieber",
    author_email="opherlie@gmail.com",
    install_requires=[
        'gym',
        'cloudpickle',
        'opencv-python',
    ],
    packages=["rltime"],
    python_requires='>=3.6',
    description="RLtime is a reinforcement learning library focused on state-of-the-art q-learning algorithms and features",
    long_description="""
RLtime is a reinforcement learning library, currently supporting PyTorch, with focus on state-of-the-art q-learning algorithms and features, and interacting with real-time environments which require low-latency acting and sample-efficient training.
The latest code and instructions can be found here:
https://github.com/opherlieber/rltime
"""
)
