from app.tools.base import ExternalToolAdapter, ToolSpec


class WebSearcherTool(ExternalToolAdapter):
    spec = ToolSpec(
        name="web_searcher",
        description="Search the public web using an external search service.",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        layer="external_tool_adapter",
    )
    status_message = "Searching the web..."
    invoke_path = "/invoke"
