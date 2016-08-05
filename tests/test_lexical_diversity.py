import pytest

import tools

# this allows using the fixture in all tests in this module
pytestmark = pytest.mark.usefixtures("spark_context")


def test_lexical_diversity(spark_context):
    test_input = "Hola Mundo"
    results = tools.lexical_diversity(test_input)
    expected_results = 0.9
    assert results == expected_results
