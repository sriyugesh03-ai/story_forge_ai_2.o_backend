import base64
import os

from dotenv import load_dotenv

load_dotenv()


def _derive_clerk_config() -> tuple[str, str]:
    """Derive Clerk issuer/JWKS/audience from the publishable key when not set explicitly.

    Clerk publishable keys look like `pk_test_<base64>` where the base64 payload is
    either `<instance_domain>$` or `<frontend_app_id>$<instance_domain>`.
    """
    publishable = os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY") or os.getenv("CLERK_PUBLISHABLE_KEY", "")
    issuer = os.getenv("CLERK_ISSUER", "")
    audience = os.getenv("CLERK_AUDIENCE", "")
    if not issuer and publishable.startswith("pk_"):
        try:
            raw_b64 = publishable.split("_", 2)[-1]  # strip "pk_test_" / "pk_live_"
            raw_b64 += "=" * (-len(raw_b64) % 4)
            raw = base64.b64decode(raw_b64).decode("utf-8")
            if raw.endswith("$"):
                domain = raw[:-1]
                issuer = f"https://{domain}"
            else:
                app_id, sep, domain = raw.partition("$")
                if domain:
                    issuer = f"https://{domain}"
                    if not audience and app_id:
                        audience = app_id
        except Exception:
            pass
    return issuer, audience


_DERIVED_ISSUER, _DERIVED_AUDIENCE = _derive_clerk_config()
_CLERK_ISSUER = os.getenv("CLERK_ISSUER", _DERIVED_ISSUER)


class Settings:
    """Application-wide configuration settings loaded from environment variables."""

    APP_NAME: str = os.getenv("APP_NAME", "Story Forge AI 2.O")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.5-flash")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "5000"))
    MANGO_DB_URL: str = os.getenv("MANGO_DB_URL")
    MONGO_DB_LOCAL: str = os.getenv("MONGO_DB_LOCAL")
    DB_NAME: str = os.getenv("DB_NAME", "rag_db")

    # ── Application tokens ────────────────────────────────────────────
    APP_JWT_SECRET: str = os.getenv("APP_JWT_SECRET", "")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

    # ── Refresh-token cookie ──────────────────────────────────────────
    COOKIE_NAME: str = os.getenv("COOKIE_NAME", "sf_refresh")
    COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "true").lower() == "true"
    COOKIE_SAMESITE: str = os.getenv("COOKIE_SAMESITE", "strict").lower()
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # ── Clerk ─────────────────────────────────────────────────────────
    CLERK_PUBLISHABLE_KEY: str = os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY") or os.getenv("CLERK_PUBLISHABLE_KEY", "")
    CLERK_SECRET_KEY: str = os.getenv("CLERK_SECRET_KEY", "")
    CLERK_ISSUER: str = _CLERK_ISSUER
    CLERK_AUDIENCE: str = os.getenv("CLERK_AUDIENCE", _DERIVED_AUDIENCE)
    CLERK_JWKS_URL: str = os.getenv(
        "CLERK_JWKS_URL",
        f"{_CLERK_ISSUER}/.well-known/jwks.json" if _CLERK_ISSUER else "",
    )

    # ── GitHub OAuth ──────────────────────────────────────────────────
    GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET", "")
    GITHUB_REDIRECT_URI: str = os.getenv("GITHUB_REDIRECT_URI", "")
    GITHUB_SCOPES: str = os.getenv("GITHUB_SCOPES", "read:user repo")
    GITHUB_TOKEN_ENCRYPTION_KEY: str = os.getenv("GITHUB_TOKEN_ENCRYPTION_KEY", "")

    # ── GitHub remote endpoints (hosted by GitHub — nothing to maintain) ──
    GITHUB_AUTH_URL: str = os.getenv("GITHUB_AUTH_URL", "https://github.com/login/oauth/authorize")
    GITHUB_TOKEN_URL: str = os.getenv("GITHUB_TOKEN_URL", "https://github.com/login/oauth/access_token")
    GITHUB_API_URL: str = os.getenv("GITHUB_API_URL", "https://api.github.com")
    GITHUB_MCP_URL: str = os.getenv("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/")

    # Comma-separated allow-list of MCP tool names the AI agent may call.
    GITHUB_MCP_ALLOWED_TOOLS: str = os.getenv(
        "GITHUB_MCP_ALLOWED_TOOLS",
        "list_repositories,get_file_contents,search_repositories,list_issues,get_issue",
    )


settings = Settings()