"""Integration connectors: MCP clients + SaaS REST abstractions."""

from app.integrations.base import (
    AuthError,
    BaseConnector,
    IntegrationError,
    NotFoundError,
    RateLimitedError,
)
from app.integrations.freshdesk import FreshdeskConnector
from app.integrations.jira import JiraConnector
from app.integrations.notion import NotionConnector
from app.integrations.zoho_crm import ZohoCRMConnector
from app.integrations.zoho_desk import ZohoDeskConnector

__all__ = [
    "AuthError",
    "BaseConnector",
    "FreshdeskConnector",
    "IntegrationError",
    "JiraConnector",
    "NotFoundError",
    "NotionConnector",
    "RateLimitedError",
    "ZohoCRMConnector",
    "ZohoDeskConnector",
]
