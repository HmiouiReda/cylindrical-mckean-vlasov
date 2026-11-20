from frozen_guard import code_equal


def test_comment_and_docstring_edits_pass():
    old = b'def f(x):\n    """old."""\n    # old\n    return x + 1\n'
    new = b'def f(x):\n    """new, longer\n    docstring."""\n    # new\n    return x + 1\n'
    assert code_equal(old, new, "a.py")


def test_code_change_fails():
    old = b"def f(x):\n    return x + 1\n"
    new = b"def f(x):\n    return x + 2\n"
    assert not code_equal(old, new, "a.py")


def test_docstring_only_body_still_compares():
    old = b'def f():\n    """only a docstring."""\n'
    new = b"def f():\n    pass\n"
    assert code_equal(old, new, "a.py")


def test_non_python_compared_byte_for_byte():
    assert code_equal(b'{"k": 1}', b'{"k": 1}', "c.json")
    assert not code_equal(b'{"k": 1}', b'{"k": 2}', "c.json")
