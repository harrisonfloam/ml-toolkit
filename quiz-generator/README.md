# quiz-generator

Generate a multiple choice quiz to encourage deep retention of research paper content.

## Setup
```
git clone https://github.com/harrisonfloam/quiz-generator.git
```

Create a conda environment from the .yml file.

```
conda env create -f environment.yml
conda activate quiz-gen
```

Update it if any packages are added.

```
conda env export --no-builds > environment.yml

```

To enable module importing, run:

```
pip install -e .
```