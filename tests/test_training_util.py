import copy
from remote_sensing_ddpm.run_downstream_tasks import safe_join_dicts


def test_safe_join_dicts():
    REFERENCE_DICT_KEY = "a"
    OTHER_KEY = "b"
    REFERENCE_DICT_VALUE = {"c": 3}
    OTHER_VALUE = 2
    assert REFERENCE_DICT_KEY != OTHER_KEY
    assert REFERENCE_DICT_VALUE != OTHER_VALUE

    # Create needed dictionaries
    reference_dict = {REFERENCE_DICT_KEY: REFERENCE_DICT_VALUE}
    disjoint_dict = {OTHER_KEY: OTHER_VALUE}
    reference_dict_ok = copy.deepcopy(reference_dict)
    reference_dict_not_ok = {REFERENCE_DICT_KEY: OTHER_VALUE}

    # These should run without error
    safe_join_dicts(dict_a=reference_dict, dict_b=reference_dict_ok)
    safe_join_dicts(
        dict_a=reference_dict, dict_b=disjoint_dict,
    )

    # These should fail
    assertion_error = False
    try:
        safe_join_dicts(dict_a=reference_dict, dict_b=reference_dict_not_ok)
    except AssertionError:
        assertion_error = True
    assert assertion_error
