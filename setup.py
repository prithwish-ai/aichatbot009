#!/usr/bin/env python3
"""
Setup script for the progress_tracker package.
"""

from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="iti-chatbot",
    version="1.0.0",
    author="ITI Chatbot Team",
    author_email="info@itichatbot.example.com",
    description="AI-powered chatbot for Industrial Training Institutes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iti-chatbot/iti-chatbot",
    project_urls={
        "Bug Tracker": "https://github.com/iti-chatbot/iti-chatbot/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "iti-chatbot=run_iti_app:main",
        ],
    },
) 