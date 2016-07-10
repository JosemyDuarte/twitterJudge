import pytest

import tools


@pytest.fixture(scope="session")
def spark_context(request):
    """ fixture for creating a spark context
    Args:
        request: pytest.FixtureRequest object
    """
    conf = (SparkConf().setMaster("local[2]").setAppName("pytest-pyspark-local-testing"))
    sc = SparkContext(conf=conf)
    request.addfinalizer(lambda: sc.stop())

    quiet_py4j()
    return sc


pytestmark = pytest.mark.usefixtures("spark_context")


def test_do_word_counts(spark_context):
    """ test word couting
    Args:
        spark_context: test fixture SparkContext
    """
    test_input = [
        ' hello spark ',
        ' hello again spark spark'
    ]

    input_rdd = spark_context.parallelize(test_input, 1)
    results = wordcount.do_word_counts(input_rdd)

    expected_results = {'hello': 2, 'spark': 3, 'again': 1}
    assert results == expected_results