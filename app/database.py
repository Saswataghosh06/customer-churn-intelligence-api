# app/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

# Create the async engine
engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)

# Create a session factory
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Base class for our ORM models
class Base(DeclarativeBase):
    pass

# Dependency to get DB sessions in our endpoints
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()