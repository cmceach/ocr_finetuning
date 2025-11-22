from setuptools import setup, find_packages

setup(
    name="ocr-finetuning",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "python-doctr[torch]>=0.9.0",
        "azure-ai-documentintelligence>=1.0.0",
        "label-studio-sdk>=1.0.0",
        "torch>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
)

