from app.tools.base import ExternalToolAdapter, ToolSpec


class FileViewerTool(ExternalToolAdapter):
    spec = ToolSpec(
        name="file_viewer",
        description="Preview or read file contents through an external file service.",
        args_schema={
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "path": {"type": "string"},
                "page": {"type": "integer"},
            },
        },
        layer="external_tool_adapter",
    )
    status_message = "Opening file..."
    invoke_path = "/invoke"
