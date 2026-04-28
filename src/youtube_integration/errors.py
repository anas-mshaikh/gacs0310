class YouTubeIntegrationError(Exception):
    """Base error for YouTube integration failures."""


class YouTubeConfigError(YouTubeIntegrationError):
    """Raised when YouTube integration configuration is invalid."""


class YouTubeSecretError(YouTubeIntegrationError):
    """Raised when YouTube OAuth secrets cannot be loaded or validated."""


class YouTubeOAuthError(YouTubeIntegrationError):
    """Raised when OAuth credential creation or refresh fails."""
