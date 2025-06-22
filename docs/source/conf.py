# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

project = 'lambdapic'
copyright = '2025, xsgeng'
author = 'xsgeng'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",    # Auto-generate docs from docstrings
    "sphinx.ext.viewcode",   # Add links to source code
    "sphinx.ext.napoleon",   # Support Google-style/Numpy docstrings
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.mermaid", # Mermaid diagram support
]



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_logo = '../../lambdaPIC.svg'
html_favicon = '../../lambdaPIC.svg'
html_title = 'λPIC Documentation'

autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
