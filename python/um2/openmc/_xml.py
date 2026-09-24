"""XML helpers (compatible with openmc._xml), using xml.etree."""

import numpy as np


def clean_indentation(element, level=0, spaces_per_level=2, trailing_indent=True):
    """Set indentation of XML element and its sub-elements."""
    i = "\n" + level*spaces_per_level*" "
    if not element.tail:
        element.tail = ""
    if len(element):
        if not element.text or not element.text.strip():
            element.text = i + spaces_per_level*" "
        if trailing_indent and (not element.tail or not element.tail.strip()):
            element.tail = i
        for sub_element in element:
            clean_indentation(sub_element, level+1, spaces_per_level)
        if not sub_element.tail or not sub_element.tail.strip():
            sub_element.tail = i
    else:
        if trailing_indent and level and (not element.tail or not element.tail.strip()):
            element.tail = i


def get_text(elem, name, default=None):
    """Get the value of an attribute or the text of a sub-element."""
    if name in elem.attrib:
        return elem.get(name, default)
    child = elem.find(name)
    return child.text if child is not None else default


def get_elem_list(elem, name, dtype=int):
    """Get a list of values from an attribute or sub-element."""
    text = get_text(elem, name)
    if text is not None:
        return [dtype(x) for x in text.split()]


def get_elem_array(elem, name, dtype=int):
    """Get a numpy array of values from an attribute or sub-element."""
    text = get_text(elem, name)
    if text is not None:
        text = text.strip()
        if not text:
            return np.array([], dtype=dtype)
        return np.array(text.split(), dtype=dtype)
