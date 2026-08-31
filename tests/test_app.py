from purchase_propensity.app import commercial_recommendation


def test_commercial_recommendations_follow_probability_tiers() -> None:
    assert commercial_recommendation(0.8)[0] == "High priority"
    assert commercial_recommendation(0.5)[0] == "Nurture opportunity"
    assert commercial_recommendation(0.1)[0] == "Low priority"
