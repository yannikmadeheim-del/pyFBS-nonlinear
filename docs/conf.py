# Configuration file for the Sphinx documentation builder.
#
# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.autosummary', 'sphinx.ext.viewcode', 'sphinx.ext.coverage',
              'sphinx.ext.doctest', 'sphinx.ext.ifconfig', 'sphinx.ext.mathjax',
              'sphinx.ext.napoleon','sphinxcontrib.bibtex','sphinx.ext.autodoc',
              'sphinx.ext.doctest','sphinx.ext.autosummary','sphinx.ext.extlinks',
              'sphinx.ext.coverage','sphinx_design',
              ]
			  
bibtex_bibfiles = ['paper.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'pyFBS'
copyright = "2026, The pyFBS Developers"
author = "The pyFBS Developers"
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyfbsdoc'

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': '',

    # Latex figure (float) alignment
    #
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'pyfbs.tex',
     'pyFBS Documentation',
     'pyFBS developers', 'manual'),
]

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'pyfbs',
     'pyFBS Documentation',
     [author], 1)
]

html_favicon = './logo/logo-small.png'

html_theme_options = {
    'logo_only': True,
    'body_max_width': '50%'
}

html_logo = "./logo/logo-big.png"

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'pyfbs',
     'pyFBS Documentation',
     author,
     'pyfbs',
     'One line description of project.',
     'Miscellaneous'),
]

def setup(app):
    app.add_css_file("style.css")
