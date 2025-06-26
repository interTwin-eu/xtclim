# CI Cheatsheet

## Objective

* Understand the purpose of CI (Continuous Integration).
* Learn how to create and structure a CI workflow on GitHub.
* Understand each step of a typical CI: linting, testing, link checking.
* Explain the tools used: `pytest`, `ruff`, `flake8`, etc.

## What is CI (Continuous Integration)

CI is a DevOps practice that automatically tests and validates code
at each modification (push or pull request) to:

* Detect errors early (through tests, linting, checks)
* Maintain code quality (enforce formatting and standards)
* Facilitate collaboration (fast feedback and clean code)

GitHub Actions allows automating these checks via workflow files defined in `.github/workflows`.

## Recommended Project Structure

Here is a typical structure for a Python project using CI:

```
my-project/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow
│   └── linters/
│       └── mlc_config.json     # Link checker configuration
├── my_project/                 # Source code
├── tests/                      # Unit tests
├── setup.py or pyproject.toml
└── requirements.txt or requirements-dev.txt
```

## File: `.github/workflows/ci.yml` Explained

```yaml
name: CI - Lint, Tests, and Link Check

# Trigger the workflow on every push or pull request
on:
  push:
  pull_request:

# Minimal permissions
permissions:
  contents: read
  actions: read

jobs:
  lint-test-links:
    name: Lint with Ruff, run tests, and check links
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install project and dev dependencies
        run: |
          python -m pip install --upgrade pip
          pip install .[dev] || pip install .
          pip install pytest pytest-cov ruff

      - name: Lint with Ruff (check and fix)
        run: |
          ruff check . --fix
          ruff format .

      - name: Lint with flake8 (SQAaaS requirement)
        run: |
          pip install flake8
          flake8 .

      - name: Run tests with coverage
        run: |
          pytest --cov=itwinai.plugins.xtclim --cov-report=xml
          coverage report --fail-under=70

      - name: Check Markdown links
        uses: gaurav-nelson/github-action-markdown-link-check@v1
        with:
          config-file: ".github/linters/mlc_config.json"
          check-modified-files-only: "yes"
          use-quiet-mode: "yes"
          use-verbose-mode: "yes"
          base-branch: "main"
```

## Tools Used in the Workflow

### pytest

A Python unit testing framework.

* Discovers and runs test files in the `tests/` folder
* Built-in support for assertions
* Works with `pytest-cov` to measure code coverage

Example:

```bash
pytest --cov=module_path
```

### ruff

A fast Python linter and formatter.

* Compatible with `flake8`, `isort`, and `black` rules
* Can automatically fix linting errors (`--fix`)
* Also handles code formatting

Examples:

```bash
ruff check . --fix
ruff format .
```

### flake8

A classic Python linter.

* Enforces PEP8 style guidelines
* Does not auto-correct, but widely supported and required by some tools like SQAaaS

Example:

```bash
flake8 .
```

### coverage

A tool to measure how much of your code is tested.

* Used in combination with `pytest-cov`

Example:

```bash
coverage report --fail-under=70
```

### markdown-link-check

A GitHub Action that checks for broken Markdown links in `.md` files.

* Configured here with `.github/linters/mlc_config.json`
* Useful to ensure documentation is clean and up to date

## Why is This CI Workflow Effective

* Uses automatic linting and formatting (`ruff`)
* Manual linting included for compatibility (`flake8`)
* Enforces a minimum test coverage threshold
* Checks documentation links
* Compliant with SQAaaS or other quality frameworks

## Going Further

* Add build/package steps
* Upload test and coverage artifacts (e.g. `dist/`, `coverage.xml`)
* Use `pre-commit` hooks to run Ruff/Flake8 locally before pushing
