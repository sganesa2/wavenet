import pytest
from ignore_file import main

@pytest.mark.parametrize(
        ["inp","expected"],
        [
            (0,1),
            (1,2),
            (2,3)
        ]
)
def test_main(inp:int, expected:int):
    assert main(inp)==expected, "Test failed for input {inp:%s}"%inp