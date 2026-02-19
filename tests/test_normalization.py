from coverage_pipeline.normalization import canonicalize_coverage_status


def test_pa_status_mapping():
    result = canonicalize_coverage_status("Covered with condition", "PA")
    assert result == "Coverage with Conditions(PA Required)"


def test_st_status_mapping():
    result = canonicalize_coverage_status("Covered with condition", "DST")
    assert result == "Coverage with Conditions(ST Required)"


def test_generic_mapping_without_acronym():
    result = canonicalize_coverage_status("Covered", "")
    assert result == "Covered"


def test_invalid_mapping_returns_none():
    result = canonicalize_coverage_status("UNKNOWN_TEXT", "QL")
    assert result is None
