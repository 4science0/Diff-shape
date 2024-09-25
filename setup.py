from setuptools import setup, find_packages

reqs=[
    ]

setup(
    name='DiffShape',
    version='0.0.1',
    url=None,
    author='Mingyuan Xu, Jie Lin',
    author_email='lin_jie@gzlab.ac.cn',
    description='Diff-Shape: A Novel Constrained Diffusion Model for Shape based De Novo Drug Design',
    packages=find_packages(exclude=["wandb", "archives", "configs"]),
    install_requires=reqs
)
