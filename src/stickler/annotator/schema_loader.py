"""Schema loading from JSON Schema files, Pydantic imports, and builder output.

Provides three entry points that all produce a consistent
``(raw_json_schema, StructuredModel_subclass)`` tuple:

- ``from_json_schema_file``: Load a ``.json`` file from disk.
- ``from_pydantic_import``: Import a ``StructuredModel`` subclass by dotted path.
- ``from_builder_schema``: Convert an in-memory schema dict (e.g. from the
  Schema Builder UI) into a model class.

All paths preserve ``x-aws-stickler-*`` extension fields as-is.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Type

from stickler.structured_object_evaluator import StructuredModel


class SchemaLoader:
    """Loads annotation schemas from various sources.

    Every static method returns a ``(raw_schema_dict, model_class)`` tuple
    where *raw_schema_dict* is the JSON Schema (with any
    ``x-aws-stickler-*`` extensions intact) and *model_class* is a
    ``StructuredModel`` subclass ready to validate annotation data.
    """

    @staticmethod
    def from_json_schema_file(path: str | Path) -> tuple[dict, Type[StructuredModel]]:
        """Load a JSON Schema file and return ``(raw_schema, model_class)``.

        Args:
            path: Filesystem path to a ``.json`` schema file.

        Returns:
            Tuple of the raw schema dict and a ``StructuredModel`` subclass.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file contains invalid JSON or an invalid schema.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read schema file {path}: {exc}") from exc

        try:
            schema = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in schema file {path}: {exc}"
            ) from exc

        if not isinstance(schema, dict):
            raise ValueError(
                f"Schema file {path} must contain a JSON object, "
                f"got {type(schema).__name__}"
            )

        try:
            model_class = StructuredModel.from_json_schema(schema)
        except (ValueError, Exception) as exc:
            raise ValueError(
                f"Invalid schema in {path}: {exc}"
            ) from exc

        return schema, model_class

    @staticmethod
    def from_pydantic_import(import_path: str) -> tuple[dict, Type[StructuredModel]]:
        """Import a ``StructuredModel`` subclass by dotted path.

        Args:
            import_path: Dotted Python import path, e.g.
                ``"mypackage.models.InvoiceModel"``.

        Returns:
            Tuple of the JSON Schema dict and the imported model class.

        Raises:
            ValueError: If *import_path* cannot be split into module + class.
            ImportError: If the module cannot be imported.
            TypeError: If the resolved class is not a ``StructuredModel``
                subclass.
        """
        if "." not in import_path:
            raise ValueError(
                f"Import path must be a dotted path like "
                f"'package.module.ClassName', got: '{import_path}'"
            )

        module_path, _, class_name = import_path.rpartition(".")

        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Could not import module '{module_path}': {exc}"
            ) from exc

        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(
                f"Module '{module_path}' has no attribute '{class_name}'"
            ) from exc

        if not (isinstance(cls, type) and issubclass(cls, StructuredModel)):
            raise TypeError(
                f"'{import_path}' is not a StructuredModel subclass "
                f"(got {type(cls).__name__}: {cls})"
            )

        schema = cls.to_json_schema()
        return schema, cls

    @staticmethod
    def from_builder_schema(schema: dict) -> tuple[dict, Type[StructuredModel]]:
        """Convert a schema builder dict to ``(schema, model_class)``.

        Args:
            schema: A raw JSON Schema dict, typically produced by the
                Schema Builder UI.

        Returns:
            Tuple of the schema dict and a ``StructuredModel`` subclass.

        Raises:
            ValueError: If the schema is invalid or cannot be converted.
        """
        if not isinstance(schema, dict):
            raise ValueError(
                f"Schema must be a dict, got {type(schema).__name__}"
            )

        try:
            model_class = StructuredModel.from_json_schema(schema)
        except (ValueError, Exception) as exc:
            raise ValueError(
                f"Invalid schema from builder: {exc}"
            ) from exc

        return schema, model_class
