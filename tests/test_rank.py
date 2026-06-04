from api import rank, RankRequest


def test_rank_returns_20_for_known_user():
    result = rank(RankRequest(user_id=1))
    assert len(result["recommendations"]) == 20
