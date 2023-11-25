import re
from setuptools import setup, find_packages


def get_version():
    filename = "PIFuHD/__init__.py"
    with open(filename) as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def get_install_requires():
    with open("requirements.txt") as req:
        return req.read().split("\n")


def get_long_description():
    with open("README.md") as f:
        return f.read()


def main():
    setup(
        name='pifu-hd',
        version=get_version(),
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        packages=find_packages(),
        url='https://github.com/facebookresearch/pifuhd',
        license='Creative Commons',
        author='Shunsuke Saito',
        author_email='shunsuke.saito16@gmail.com',
        description=' PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization (CVPR 2020)',
        install_requires=get_install_requires(),
        package_data={"PIFuHD": []},
        entry_points={"console_scripts": ["pifuhd=PIFuHD.__main__:main"]},
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3 :: Only",
        ],
    )


if __name__ == "__main__":
    main()
