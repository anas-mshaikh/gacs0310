# YouTube OAuth Setup

## Purpose

Phase 2 Challenge 2 requires YouTube analytics collection for GACS-generated videos.

The YouTube Analytics API requires OAuth 2.0 authorization. API keys are not sufficient for analytics reports.

## Required Scopes

- `https://www.googleapis.com/auth/youtube.readonly`
- `https://www.googleapis.com/auth/yt-analytics.readonly`

## Secret Payload

Store the OAuth payload in the deployment secret manager.

```json
{
  "client_id": "...",
  "client_secret": "...",
  "refresh_token": "...",
  "token_uri": "https://oauth2.googleapis.com/token",
  "scopes": [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly"
  ]
}
```
