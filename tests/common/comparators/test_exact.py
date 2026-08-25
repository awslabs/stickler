"""Tests for ExactComparator."""

from stickler.comparators import ExactComparator


class TestExactComparator:
    """Test the ExactComparator.

    As of 0.7.0, ExactComparator is truly exact:
    - Default is case_sensitive=True
    - No punctuation/whitespace normalization
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Default: case-sensitive, no normalization
        self.comparator = ExactComparator()
        # Case-insensitive variant
        self.case_insensitive = ExactComparator(case_sensitive=False)

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        assert self.comparator.compare("hello", "hello") == 1.0
        assert self.comparator.compare("123", "123") == 1.0
        assert self.comparator.compare("", "") == 1.0
        assert self.comparator.compare("SHP-2024-001", "SHP-2024-001") == 1.0

    def test_case_sensitive_by_default(self):
        """Test that case matters by default (the #199 fix)."""
        # Default is now case-sensitive
        assert self.comparator.compare("Hello", "hello") == 0.0
        assert self.comparator.compare("WORLD", "world") == 0.0
        assert self.comparator.compare("SHP-2024-001", "shp-2024-001") == 0.0

    def test_case_insensitive_option(self):
        """Test that case_sensitive=False uses casefold for matching."""
        assert self.case_insensitive.compare("Hello", "hello") == 1.0
        assert self.case_insensitive.compare("WORLD", "world") == 1.0
        assert self.case_insensitive.compare("Hello World", "hello world") == 1.0
        # casefold handles German ß correctly
        assert self.case_insensitive.compare("STRASSE", "straße") == 1.0

    def test_no_punctuation_normalization(self):
        """Test that punctuation is NOT stripped (the #199 fix).

        This is the key behavior change: "SHP-2024-001" must NOT match
        "shp 2024 001" because they are different strings.
        """
        # These should NOT match - punctuation matters
        assert self.comparator.compare("hello, world", "hello world") == 0.0
        assert self.comparator.compare("hello-world", "hello world") == 0.0
        assert self.comparator.compare("hello.world", "hello world") == 0.0
        assert self.comparator.compare("hello  world", "hello world") == 0.0
        # The #199 bug case
        assert self.comparator.compare("SHP-2024-001", "SHP 2024 001") == 0.0
        assert self.comparator.compare("ID-123", "ID 123") == 0.0

    def test_non_matching(self):
        """Test that non-matching strings return 0.0."""
        assert self.comparator.compare("hello", "world") == 0.0
        assert self.comparator.compare("123", "456") == 0.0
        assert self.comparator.compare("hello", "hello world") == 0.0

    def test_none_values(self):
        """Test that None values are handled correctly."""
        assert self.comparator.compare(None, None) == 1.0
        assert self.comparator.compare("hello", None) == 0.0
        assert self.comparator.compare(None, "world") == 0.0

    def test_non_string_values(self):
        """Test that non-string values are converted to strings."""
        assert self.comparator.compare(123, "123") == 1.0
        assert self.comparator.compare("456", 456) == 1.0
        assert self.comparator.compare(789, 789) == 1.0

    def test_binary_compare_match(self):
        """Test that binary_compare returns (1, 0) for match."""
        assert self.comparator.binary_compare("hello", "hello") == (1, 0)
        # Case mismatch is now a non-match by default
        assert self.comparator.binary_compare("Hello", "hello") == (0, 1)
        # Case-insensitive comparator matches
        assert self.case_insensitive.binary_compare("Hello", "hello") == (1, 0)

    def test_binary_compare_no_match(self):
        """Test that binary_compare returns (0, 1) for no match."""
        assert self.comparator.binary_compare("hello", "world") == (0, 1)
        assert self.comparator.binary_compare("Hello", "hello") == (0, 1)

    def test_custom_threshold(self):
        """Test that the threshold is applied correctly.

        ExactComparator only returns 0.0 or 1.0, so thresholds below 1.0
        effectively treat any non-null pair as a match in binary_compare.
        """
        comparator = ExactComparator(threshold=0.5)
        assert comparator.binary_compare("hello", "hello") == (1, 0)
        assert comparator.binary_compare("hello", "world") == (0, 1)

    def test_config_property(self):
        """Test that config only includes non-default values."""
        # Default config is None (all defaults)
        assert ExactComparator().config is None
        assert ExactComparator(case_sensitive=True).config is None

        # Non-default config is serialized
        assert ExactComparator(case_sensitive=False).config == {"case_sensitive": False}

    def test_name_property(self):
        """Test the name property."""
        assert self.comparator.name == "exact"

    def test_repr(self):
        """Test string representation."""
        assert "threshold=1.0" in repr(self.comparator)
        assert "case_sensitive=False" not in repr(self.comparator)
        assert "case_sensitive=False" in repr(self.case_insensitive)


class TestExactComparatorIdentifiers:
    """Test ExactComparator with real-world identifier patterns.

    This class specifically tests the #199 fix: identifiers must match exactly.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = ExactComparator()

    def test_invoice_ids(self):
        """Invoice IDs must match exactly."""
        assert self.comparator.compare("INV-2024-001", "INV-2024-001") == 1.0
        assert self.comparator.compare("INV-2024-001", "inv-2024-001") == 0.0
        assert self.comparator.compare("INV-2024-001", "INV 2024 001") == 0.0

    def test_uuids(self):
        """UUIDs must match exactly."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert self.comparator.compare(uuid, uuid) == 1.0
        assert self.comparator.compare(uuid, uuid.upper()) == 0.0
        assert self.comparator.compare(uuid, uuid.replace("-", "")) == 0.0

    def test_skus(self):
        """SKUs must match exactly."""
        assert self.comparator.compare("SKU-ABC-123", "SKU-ABC-123") == 1.0
        assert self.comparator.compare("SKU-ABC-123", "sku-abc-123") == 0.0
        assert self.comparator.compare("SKU-ABC-123", "SKU ABC 123") == 0.0

    def test_phone_numbers(self):
        """Phone numbers: different formats are different strings.

        Note: For phone number semantic equivalence, use a specialized
        comparator or pre-normalize to E.164 format.
        """
        assert self.comparator.compare("555-123-4567", "555-123-4567") == 1.0
        assert self.comparator.compare("555-123-4567", "(555) 123-4567") == 0.0
        assert self.comparator.compare("555-123-4567", "5551234567") == 0.0

    def test_emails(self):
        """Emails: ExactComparator is case-sensitive by default.

        Note: For RFC-compliant email matching (domain case-insensitive),
        use case_sensitive=False or a specialized email comparator.
        """
        assert self.comparator.compare("user@example.com", "user@example.com") == 1.0
        assert self.comparator.compare("user@example.com", "USER@EXAMPLE.COM") == 0.0

        # Case-insensitive option for emails
        ci = ExactComparator(case_sensitive=False)
        assert ci.compare("user@example.com", "USER@EXAMPLE.COM") == 1.0
