from setuptools import find_packages, setup

setup(
    name='rlsp',
    version=1.0,
    description='Reward Learning by Simulating the Past',
    author='Rohin Shah, Dmitrii Krasheninnikov, Jordan Alexander, et al',
    author_email='rohinmshah@berkeley.edu',
    python_requires='>=3.6.0',
    url='https://github.com/HumanCompatibleAI/rlsp',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.13',
        'scipy>=0.19',
    ],
    test_suite='nose.collector',
    tests_require=['nose', 'nose-cover3'],
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
