import unittest

from weather_strategy.http import _redact_url


class HttpTest(unittest.TestCase):
    def test_redact_url_hides_common_secret_query_params(self) -> None:
        redacted = _redact_url(
            "https://api.example.com/forecast?api_key=expensive-secret&city=NYC&access-token=token-secret"
        )

        self.assertNotIn("expensive-secret", redacted)
        self.assertNotIn("token-secret", redacted)
        self.assertIn("api_key=REDACTED", redacted)
        self.assertIn("city=NYC", redacted)
        self.assertIn("access-token=REDACTED", redacted)


if __name__ == "__main__":
    unittest.main()
