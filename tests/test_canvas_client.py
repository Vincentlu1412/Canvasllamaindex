"""Tests for the Canvas client's pure logic.

Everything network-touching is left untested here on purpose -- these run
in CI with no Canvas instance and no token. What's covered is the parts
that have actually bitten: pagination header parsing, HTML stripping, and
the config guards that keep a bearer token from being sent somewhere it
shouldn't.
"""

from __future__ import annotations

import pytest

from canvas_client import CanvasClient, CanvasNotConfigured, _next_page_url, _strip_html


class TestNextPageUrl:
    def test_returns_none_without_header(self) -> None:
        assert _next_page_url(None) is None

    def test_returns_none_when_no_next_rel(self) -> None:
        header = '<https://x.instructure.com/api/v1/courses?page=1>; rel="current"'
        assert _next_page_url(header) is None

    def test_extracts_next_url(self) -> None:
        header = (
            '<https://x.instructure.com/api/v1/courses?page=1>; rel="current",'
            '<https://x.instructure.com/api/v1/courses?page=2>; rel="next",'
            '<https://x.instructure.com/api/v1/courses?page=9>; rel="last"'
        )
        assert _next_page_url(header) == "https://x.instructure.com/api/v1/courses?page=2"


class TestStripHtml:
    def test_removes_tags_and_collapses_whitespace(self) -> None:
        assert _strip_html("<p>Midterm  is\n<b>Friday</b></p>") == "Midterm is Friday"

    def test_handles_empty_input(self) -> None:
        assert _strip_html("") == ""


class TestFromEnv:
    def test_raises_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CANVAS_API_TOKEN", raising=False)
        monkeypatch.setenv("CANVAS_BASE_URL", "https://x.instructure.com")
        with pytest.raises(CanvasNotConfigured, match="CANVAS_API_TOKEN"):
            CanvasClient.from_env()

    def test_raises_without_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CANVAS_API_TOKEN", "fake-token-for-testing")
        monkeypatch.delenv("CANVAS_BASE_URL", raising=False)
        with pytest.raises(CanvasNotConfigured, match="CANVAS_BASE_URL"):
            CanvasClient.from_env()

    def test_rejects_plaintext_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bearer token over http:// would be sniffable in transit."""
        monkeypatch.setenv("CANVAS_API_TOKEN", "fake-token-for-testing")
        monkeypatch.setenv("CANVAS_BASE_URL", "http://x.instructure.com")
        with pytest.raises(CanvasNotConfigured, match="https"):
            CanvasClient.from_env()

    def test_strips_trailing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CANVAS_API_TOKEN", "fake-token-for-testing")
        monkeypatch.setenv("CANVAS_BASE_URL", "https://x.instructure.com/")
        assert CanvasClient.from_env().base_url == "https://x.instructure.com"
