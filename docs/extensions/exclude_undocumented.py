# extensions/exclude_undocumented.py

import sphinx


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring)
    return {"version": sphinx.__display_version__, "parallel_read_safe": True}


def has_docstring(obj, obj_name):
    docstring = getattr(obj, "__doc__", None)
    return docstring is not None


def process_docstring(app, what, name, obj, options, lines):
    if what == "function":
        if not has_docstring(obj, name):
            return False
