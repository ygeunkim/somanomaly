import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Online Som Detector",
    version="0.0.1",
    author="Young Geun Kim",
    author_email="dudrms33@g.skku.edu",
    description="Anomaly detection using self organizing maps",
    long_description=long_description,
    url="https://github.com/ygeunkim/onlinesom",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Computer Science/Statistics"
    ],
    keywords="Self Organizing Maps",
    install_requires=["numpy", "scipy"]
)