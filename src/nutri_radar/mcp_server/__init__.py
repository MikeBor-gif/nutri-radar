"""Точка входа MCP: инструменты агента наружу через MCP Python SDK (M7).

Логики здесь нет — обёртка над реестром инструментов (`ARCHITECTURE.md`).
"""

from nutri_radar.mcp_server.server import create_server, main

__all__ = ["create_server", "main"]
