from setuptools import setup, find_packages

setup(
    name='sec_helper_package',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,  # This includes non-code files defined in MANIFEST.in
    package_data={
        'sec_helper': ['data/*.json'],  # Include the JSON files in the data folder
    },
    install_requires=[
        'requests',
        'pandas',
        'scrapy',
        'statsmodels',
        'gensim',
        'scikit-learn'
    ],
    author='Ralf Kellner',
    author_email='ralf.kellner@uni-passau.de',
    description='A package to interact with SEC filings and retrieve company data.',
    url='https://github.com/yourusername/sec_helper_package',  # Replace with your actual URL if applicable
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)