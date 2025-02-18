# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2022

# Contributing Authors (alphabetical):
# Brandon Edwards (Intel)
# Patrick Foley (Intel)
# Sarthak Pati (Intel)
# Micah Sheller (Intel)


"""This package includes dependencies of the fets project."""

from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='fets_challenge',
    version='2.0',
    author='Sarthak Pati and Ujjwal Baid and Maximilian Zenk and Brandon Edwards and Micah Sheller and G. Anthony Reina and Patrick Foley and Alexey Gruzdev and Jason Martin and Shadi Albarqouni and Yong Chen and Russell Taki Shinohara and Annika Reinke and David Zimmerer and John B. Freymann and Justin S. Kirby and Christos Davatzikos and Rivka R. Colen and Aikaterini Kotrotsou and Daniel Marcus and Mikhail Milchenko and Arash Nazer and Hassan Fathallah-Shaykh and Roland Wiest Andras Jakab and Marc-Andre Weber and Abhishek Mahajan and Lena Maier-Hein and Jens Kleesiek and Bjoern Menze and Klaus Maier-Hein and Spyridon Bakas',
    description='FeTS Challenge Part 1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/FETS-AI/Challenge',
    packages=[
        'fets_challenge',
        'openfl-workspace',
    ],
    include_package_data=True,
    install_requires=[
        'openfl @ git+https://github.com/securefederatedai/openfl.git@6bbf9b62f97f50a06a9956eefacebf6d0a6cba4e',
        'GANDLF @ git+https://github.com/CBICA/GaNDLF.git@4d614fe1de550ea4035b543b4c712ad564248106',
        'fets @ git+https://github.com/FETS-AI/Algorithms.git@fets_challenge',
    ],
    python_requires='>=3.10, <3.13',
    classifiers=[
        'Environment :: Console',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 5 - Production/Stable',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        # Pick your license as you wish
        'License :: OSI Approved :: FETS UI License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ]
)
