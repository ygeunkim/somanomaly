import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SomAnomaly",
    version="0.0.3",
    author="Young Geun Kim",
    author_email="dudrms33@g.skku.edu",
    description="Anomaly detection using self-organizing maps",
    long_description=long_description,
    url="https://github.com/ygeunkim/somanomaly",
    license="LICENSE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"
    ],
    keywords=[
        "Self-Organizing Maps",
        "Anomaly Detection"
    ],
    install_requires=["numpy", "scipy", "pandas", "plotly", "scikit-learn", "tqdm"]
)
