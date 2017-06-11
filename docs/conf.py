# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os, sys

import mock

MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'pandas', 'attrs', 'cython', 'tables']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

source_suffix = '.rst'
master_doc = 'index'
project = 'PySight'
year = '2017'
author = 'Hagai Hargil'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.5.9'
autodoc = True

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/HagaiHargil/python-pysight/issues/%s', '#'),
    'pr': ('https://github.com/HagaiHargil/python-pysight/pull/%s', 'PR #'),
}
import sphinx_py3doc_enhanced_theme
html_theme = "sphinx_py3doc_enhanced_theme"
html_theme_path = [sphinx_py3doc_enhanced_theme.get_html_theme_path()]
html_theme_options = {
    'githuburl': 'https://github.com/HagaiHargil/python-pysight/'
}

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_use_index = True
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
