"""Tests for the rate limiter."""

from resume_chatbot.rate_limit import RateLimiter


def test_allows_requests_under_limit():
    limiter = RateLimiter(per_ip_limit=3, daily_limit=10)
    assert limiter.check("1.2.3.4") is None
    assert limiter.check("1.2.3.4") is None
    assert limiter.check("1.2.3.4") is None


def test_blocks_per_ip_over_limit():
    limiter = RateLimiter(per_ip_limit=2, daily_limit=100)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")
    result = limiter.check("1.2.3.4")
    assert result is not None
    assert "Rate limit" in result


def test_different_ips_have_separate_limits():
    limiter = RateLimiter(per_ip_limit=1, daily_limit=100)
    assert limiter.check("1.1.1.1") is None
    assert limiter.check("2.2.2.2") is None
    # First IP is now blocked
    assert limiter.check("1.1.1.1") is not None


def test_global_daily_cap():
    limiter = RateLimiter(per_ip_limit=100, daily_limit=3)
    limiter.check("1.1.1.1")
    limiter.check("2.2.2.2")
    limiter.check("3.3.3.3")
    result = limiter.check("4.4.4.4")
    assert result is not None
    assert "Daily limit" in result


def test_daily_count_tracks():
    limiter = RateLimiter(per_ip_limit=100, daily_limit=100)
    assert limiter.daily_count == 0
    limiter.check("1.1.1.1")
    limiter.check("1.1.1.1")
    assert limiter.daily_count == 2
