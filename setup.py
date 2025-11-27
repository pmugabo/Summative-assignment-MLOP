from setuptools import setup, find_packages

setup(
    name="urbansound_classification",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.6.0",
        "numpy",
        "librosa",
        "soundfile",
        "pydub",
        "joblib",
        "scikit-learn",
        "matplotlib",
    ],
)