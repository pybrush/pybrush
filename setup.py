from setuptools import setup
setup(
    name='pybrush',
    license='MIT License',
    packages=['pybrush'],
    install_requires=['numpy', 'scipy'],
    extras_require={'tests': ['pytest', 'pytest-mpl']},
    py_modules=['pybrush'],
    version='0.5.2',  # note to self: also update the one is the source!
    description='A Python Machine Unlearning Framework.',
    author='Snehil Kumar',
    author_email='snehil03july@gmail.com',
    url='https://github.com/pybrush/pybrush',
    #download_url='https://github.com/pybrush/pybrush',
    keywords=['machine unlearning', 'forget', 'privacy', 'data science', 'ai', 'python',
              'numpy'],
    classifiers=[]
)
