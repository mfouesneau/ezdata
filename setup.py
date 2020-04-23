from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name="ezdata",
      version=0.98,
      description="A Sandbox for simplistic column based data framework",
      long_description=readme(),
      author="Morgan Fouesneau",
      author_email="",
      url="https://github.com/mfouesneau/ezdata",
      packages=find_packages(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      zip_safe=False)
