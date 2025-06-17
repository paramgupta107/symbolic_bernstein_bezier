from setuptools import setup, find_packages

setup(
    name='symbolic_bernstein_bezier',
    version='0.1.1',
    author='Param Gupta',
    author_email='p.gupta@ufl.edu',
    description='Bernstein Bezier operations using SymPy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/paramgupta107/symbolic_bernstein_bezier',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'sympy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)