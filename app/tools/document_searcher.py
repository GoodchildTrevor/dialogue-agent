from app.tools.base import ExternalToolAdapter, ToolSpec


class DocumentSearcherTool(ExternalToolAdapter):
    spec = ToolSpec(
        name="document_searcher",
        description="Search corporate documents via an external retrieval API.",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        layer="external_tool_adapter",
    )
    status_message = "Searching documents..."
    # TODO: align endpoint with the real document_searcher contract.
    invoke_path = "/invoke"
