from setuptools import setup, find_packages

setup(
    name="tampering_detection",
    version="0.1.0",
    author="Priyanshu Kumar",
    author_email="priyanshuok146@yahoo.com",
    description="AI-based digital forensic image tampering detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "torch",
        "torchvision",
        "scikit-learn",
        "matplotlib",
        "Pillow",
        "tqdm",
        "grad-cam",
        "Flask",
        "fastapi",
        "uvicorn",
        "python-multipart",
    ],
    python_requires=">=3.10",
)