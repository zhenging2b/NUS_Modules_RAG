from typing import Annotated

from typing_extensions import TypedDict

MODULE_CODE_REGEX = r"[a-zA-Z]{2,3}\d{4}[a-zA-Z]?"

class ModuleQuery(TypedDict):
    """Structured schema for module search or comparison."""
    moduleCodes: Annotated[
        list[str],
        ...,
        f"List of module codes matching the pattern {MODULE_CODE_REGEX},  mentioned in the query, if any"
    ]