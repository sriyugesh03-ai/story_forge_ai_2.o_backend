import logging
from app.repo.auth_repo import AuthRepository
from app.core.security import create_access_token

logger = logging.getLogger(__name__)


class AuthService:
    """Service layer containing all authentication business logic."""

    def __init__(self, repo: AuthRepository | None = None):
        self.repo = repo or AuthRepository()

    async def register(self, username: str, email: str, password: str) -> dict:
        """Register a new user account."""
        # Validate unique username
        existing_username = await self.repo.find_by_username(username)
        if existing_username:
            raise ValueError("Username is already registered")

        # Validate unique email
        existing_email = await self.repo.find_by_email(email)
        if existing_email:
            raise ValueError("Email address is already registered")

        # Create user document
        user_data = {
            "username": username,
            "email": email,
            "password": password
        }

        await self.repo.create_user(user_data)
        logger.info(f"User '{username}' registered successfully.")

        return {"message": "Account created successfully. You can now log in."}

    async def login(self, username_or_email: str, password: str) -> dict:
        """Authenticate user credentials and issue JWT access token."""
        user = await self.repo.find_by_username_or_email(username_or_email)
        if not user or user.get("password") != password:
            raise ValueError("Invalid username or password")

        # Generate JWT token
        token = create_access_token(data={"sub": user["username"], "email": user["email"]})

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "username": user["username"],
                "email": user["email"],
                "joined": user.get("created_at")
            }
        }

    async def get_user_profile(self, username: str) -> dict:
        """Fetch sanitized user profile details."""
        user = await self.repo.find_by_username(username)
        if not user:
            raise ValueError("Authenticated user record does not exist")

        return {
            "id": str(user.get("_id", "")),
            "username": user["username"],
            "email": user["email"],
            "created_at": user.get("created_at")
        }
