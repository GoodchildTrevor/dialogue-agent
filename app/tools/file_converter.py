from app.tools.base import ExternalToolAdapter, ToolSpec


class FileConverterTool(ExternalToolAdapter):
    spec = ToolSpec(
        name="file_converter",
        description="Convert files between supported formats through an external conversion API.",
        args_schema={
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "target_format": {"type": "string"},
            },
            "required": ["source_path", "target_format"],
        },
        layer="external_tool_adapter",
    )
    status_message = "Converting file..."
    invoke_path = "/invoke"
