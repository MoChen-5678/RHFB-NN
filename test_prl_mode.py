from dpl_rhf.tests.test_prl_mode import *  # noqa: F401,F403


if __name__ == "__main__":
    import inspect

    from dpl_rhf.tests import test_prl_mode as suite

    tests = [
        function
        for name, function in inspect.getmembers(suite, inspect.isfunction)
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} strict PRL/RMF tests")
