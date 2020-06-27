import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepcut2realtime", # Replace with your own username
    version="0.0.1",
    author="Brandon Forys",
    author_email="brandon.forys@alumni.ubc.ca",
    description="An add-on for DeepLabCut that enables real-time tracking and reinforcement of animal behaviours.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bf777/DeepCut2RealTime",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)