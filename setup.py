from pip.req import parse_requirements
from setuptools import setup, find_packages

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements(<requirements_path>)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name = "ppo-ae",
    version = "0.0.1",
    description = (""),
    packages=find_packages(),
    install_requires=reqs,
)