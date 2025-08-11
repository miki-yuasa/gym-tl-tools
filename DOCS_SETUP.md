# Documentation Setup Summary

This document summarizes the documentation setup for gym-tl-tools.

## What Has Been Set Up

### 1. Sphinx Documentation Framework

- **Configuration**: `docs/conf.py` with proper settings for Python autodoc
- **Theme**: Read the Docs theme (`sphinx_rtd_theme`)
- **Extensions**: 
  - `sphinx.ext.autodoc` - Automatic documentation from docstrings
  - `sphinx.ext.napoleon` - Support for Google/NumPy style docstrings
  - `sphinx.ext.viewcode` - Add source code links
  - `sphinx.ext.intersphinx` - Link to external documentation
  - `sphinx.ext.autosummary` - Generate summary tables
  - `sphinx.ext.githubpages` - GitHub Pages support

### 2. Documentation Structure

- `docs/index.rst` - Main documentation page
- `docs/installation.rst` - Installation instructions
- `docs/quickstart.rst` - Quick start guide
- `docs/api.rst` - API reference (auto-generated from docstrings)
- `docs/examples.rst` - Detailed examples

### 3. GitHub Actions Workflow

- **File**: `.github/workflows/docs.yml`
- **Triggers**: Pushes to main branch, pull requests
- **Actions**:
  - Builds documentation using UV
  - Deploys to GitHub Pages automatically
  - Uses proper caching for faster builds

### 4. Read the Docs Configuration

- **File**: `.readthedocs.yml`
- **Features**: 
  - Python 3.11 environment
  - Automatic dependency installation
  - Sphinx HTML builder

### 5. Build Scripts and Tools

- `build_docs.sh` - Local documentation build script
- `docs/Makefile` - Sphinx makefile for various build targets
- `docs/requirements.txt` - Documentation dependencies

## How to Use

### Building Locally

```bash
# Option 1: Use the build script
./build_docs.sh

# Option 2: Use Make
cd docs && make html

# Option 3: Direct sphinx command
cd docs && uv run sphinx-build -b html . _build/html
```

### Viewing Documentation

- **Local**: Open `docs/_build/html/index.html` in browser
- **Online**: Will be available at `https://miki-yuasa.github.io/gym-tl-tools/` after GitHub Pages is enabled

### Enabling GitHub Pages

1. Go to your repository settings on GitHub
2. Navigate to "Pages" section
3. Set source to "GitHub Actions"
4. The documentation will build and deploy automatically on each push to main

### Enabling Read the Docs

1. Go to https://readthedocs.org/
2. Import your GitHub repository
3. The documentation will build automatically using the `.readthedocs.yml` configuration

## Troubleshooting

### Common Issues

1. **Import Errors**: The configuration includes mocks for dependencies that can't be installed (like `spot`)
2. **Build Warnings**: Many warnings are from docstring formatting - these don't prevent successful builds
3. **GitHub Actions Permissions**: Make sure repository has Pages write permissions enabled

### Syntax Warnings

There are regex escape sequence warnings in the source code. To fix these, update the regex patterns to use raw strings:

```python
# Instead of:
re.findall("State: (\d).*\[", raw_state)

# Use:
re.findall(r"State: (\d).*\[", raw_state)
```

## Dependencies

### Core Documentation Dependencies
- `sphinx>=7.0`
- `sphinx-rtd-theme>=2.0`  
- `sphinx-autodoc-typehints>=1.25`

These are now managed via `uv sync --group dev`

### Project Dependencies (for autodoc)
- `numpy>=1.21`
- `gymnasium>=0.26`
- `pydantic>=2.0`

## File Structure

```
gym-tl-tools/
├── docs/
│   ├── _static/           # Static files (CSS, images)
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Main page
│   ├── installation.rst  # Installation guide
│   ├── quickstart.rst    # Quick start guide
│   ├── api.rst          # API reference
│   ├── examples.rst     # Examples
│   ├── requirements.txt # Doc dependencies
│   └── Makefile         # Build commands
├── .github/workflows/
│   └── docs.yml          # GitHub Actions workflow
├── .readthedocs.yml      # Read the Docs config
└── build_docs.sh         # Local build script
```

## Next Steps

1. **Enable GitHub Pages** in repository settings
2. **Test the workflow** by pushing changes to main branch
3. **Customize styling** by adding CSS to `docs/_static/`
4. **Add more examples** as the project grows
5. **Consider versioning** for releases (Sphinx supports versioned docs)

The documentation setup is now complete and ready for use!
