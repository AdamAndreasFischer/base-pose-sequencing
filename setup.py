from setuptools import setup, find_packages

setup(
    name="base_pose_sequencing",
    version="0.1",
    packages=[package for package in find_packages()
              if package.startswith('base_pose_sequencing')],
    install_requires=[
        # Add your dependencies here
    ],
)