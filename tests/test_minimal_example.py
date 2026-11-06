import runpy


def test_minimal_example_runs(capsys):
    runpy.run_path("examples/minimal.py", run_name="__main__")
    output = capsys.readouterr().out
    assert "formula:" in output
    assert "success: True" in output
