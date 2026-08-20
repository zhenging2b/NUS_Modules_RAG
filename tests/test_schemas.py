import re

from models.schemas import MODULE_CODE_REGEX


def test_module_code_regex_matches_valid_code():
    assert re.fullmatch(MODULE_CODE_REGEX, "CS1010") is not None


def test_module_code_regex_matches_code_with_letter_suffix():
    assert re.fullmatch(MODULE_CODE_REGEX, "CS2103T") is not None


def test_module_code_regex_lowercase():
    assert re.fullmatch(MODULE_CODE_REGEX, "cs1010") is not None

def test_invalid_module_code_regex():
    assert re.fullmatch(MODULE_CODE_REGEX, "CS10") is None
    assert re.fullmatch(MODULE_CODE_REGEX, "CS10101") is None
    assert re.fullmatch(MODULE_CODE_REGEX, "CS1010TT") is None
    assert re.fullmatch(MODULE_CODE_REGEX, "C1010") is None
    assert re.fullmatch(MODULE_CODE_REGEX, "CS1010!") is None
