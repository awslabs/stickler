"""
Tests for VisualizationEngine class.
"""

from unittest.mock import patch

from stickler.reporting.html.report_config import ReportConfig
from stickler.reporting.html.visualization_engine import VisualizationEngine


class TestVisualizationEngine:
    """Test cases for VisualizationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.viz_engine = VisualizationEngine()
    
    def test_initialization(self):
        """Test VisualizationEngine initialization."""
        engine = VisualizationEngine()
        assert engine is not None
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_performance_gauge(self, mock_color_utils):
        """Test performance gauge generation."""
        mock_color_utils.return_value = "#28a745"
        
        config = ReportConfig()
        result = self.viz_engine.generate_performance_gauge(0.85, config)
        
        assert '<div class="performance-gauge">' in result
        assert '<div class="gauge-circle"' in result
        assert '85%' in result
        assert 'Overall' in result
        assert '#28a745' in result
        mock_color_utils.assert_called_once_with(0.85, config.color_thresholds)
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_performance_gauge_zero_score(self, mock_color_utils):
        """Test performance gauge with zero score."""
        mock_color_utils.return_value = "#dc3545"
        
        config = ReportConfig()
        result = self.viz_engine.generate_performance_gauge(0.0, config)
        
        assert '0%' in result
        assert '#dc3545' in result
        mock_color_utils.assert_called_once_with(0.0, config.color_thresholds)
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_performance_gauge_perfect_score(self, mock_color_utils):
        """Test performance gauge with perfect score."""
        mock_color_utils.return_value = "#28a745"
        
        config = ReportConfig()
        result = self.viz_engine.generate_performance_gauge(1.0, config)
        
        assert '100%' in result
        assert '#28a745' in result
        mock_color_utils.assert_called_once_with(1.0, config.color_thresholds)
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_field_performance_chart(self, mock_color_utils):
        """Test field performance chart generation."""
        mock_color_utils.return_value = "#ffc107"
        
        field_metrics = {
            "name": {"cm_f1": 0.85, "cm_precision": 0.90, "cm_recall": 0.80},
            "price": {"cm_f1": 0.95, "cm_precision": 0.98, "cm_recall": 0.92}
        }
        
        config = ReportConfig()
        result = self.viz_engine.generate_field_performance_chart(field_metrics, config)
        
        assert '<div class="field-chart">' in result
        assert '<h4 style="margin-bottom: 15px; color: #495057; font-size: 1.1em;">F1 Score</h4>' in result
        assert 'name' in result
        assert 'price' in result
        assert '0.850' in result
        assert '0.950' in result
        assert 'width: 85%' in result
        assert 'width: 95%' in result
        assert mock_color_utils.call_count == 2
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_field_performance_chart_with_f1_fallback(self, mock_color_utils):
        """Test field performance chart with f1 fallback key."""
        mock_color_utils.return_value = "#17a2b8"
        
        field_metrics = {
            "category": {"f1": 0.75, "precision": 0.80, "recall": 0.70}  # Uses 'f1' instead of 'cm_f1'
        }
        
        config = ReportConfig()
        result = self.viz_engine.generate_field_performance_chart(field_metrics, config)
        
        assert 'category' in result
        assert '0.750' in result
        assert 'width: 75%' in result
        mock_color_utils.assert_called_once_with(0.75, config.color_thresholds)
    
    def test_generate_field_performance_chart_escapes_field_names(self):
        """The chart escapes field names, so a later edit cannot quietly undo it (#422)."""
        field_metrics = {"<b>vendor</b> & co": {"f1": 1.0}}

        result = self.viz_engine.generate_field_performance_chart(field_metrics, ReportConfig())

        assert '<div class="field-label">&lt;b&gt;vendor&lt;/b&gt; &amp; co</div>' in result
        assert "<b>vendor</b>" not in result

    def test_generate_field_performance_chart_empty_metrics(self):
        """Test field performance chart with empty metrics."""
        field_metrics = {}
        
        config = ReportConfig()
        result = self.viz_engine.generate_field_performance_chart(field_metrics, config)
        
        assert '<div class="field-chart">' in result
        assert '<h4 style="margin-bottom: 15px; color: #495057; font-size: 1.1em;">F1 Score</h4>' in result
        assert '</div>' in result
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_field_performance_table(self, mock_color_utils):
        """Test field performance table generation."""
        mock_color_utils.return_value = "#28a745"
        
        field_metrics = {
            "name": {
                "cm_precision": 0.90,
                "cm_recall": 0.80,
                "cm_f1": 0.85,
                "tp": 45,
                "fd": 3,
                "fa": 2,
                "fn": 5
            },
            "price": {
                "cm_precision": 0.98,
                "cm_recall": 0.92,
                "cm_f1": 0.95,
                "tp": 50,
                "fd": 1,
                "fa": 1,
                "fn": 3
            }
        }
        
        config = ReportConfig()
        result = self.viz_engine.generate_field_performance_table(field_metrics, config)
        
        assert '<table class="data-table data-table-numeric" id="performance-table">' in result
        assert '<th>Field</th>' in result
        assert '<th>Precision</th>' in result
        assert '<th>Recall</th>' in result
        assert '<th>F1 Score</th>' in result
        assert '<th>TP</th>' in result
        assert '<th>FD</th>' in result
        assert '<th>FA</th>' in result
        assert '<th>FN</th>' in result
        
        # Check data rows
        assert 'name' in result
        assert 'price' in result
        assert '0.900' in result  # precision
        assert '0.800' in result  # recall
        assert '0.850' in result  # f1
        assert '45' in result     # tp
        assert 'background-color: #28a745' in result
        assert 'color: white' in result
        assert 'font-weight: bold' in result
        
        assert mock_color_utils.call_count == 2
    
    def test_generate_field_performance_table_escapes_field_names(self):
        """A field name is text, not markup, as in the chart (#422)."""
        field_metrics = {"<b>vendor</b> & co": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}

        result = self.viz_engine.generate_field_performance_table(field_metrics, ReportConfig())

        assert "<td>&lt;b&gt;vendor&lt;/b&gt; &amp; co</td>" in result
        assert "<b>vendor</b>" not in result

    def test_field_name_that_is_not_a_string_does_not_raise(self):
        """A hand-built results dict can key a field by a non-``str``."""
        field_metrics = {7: {"precision": 1.0, "recall": 1.0, "f1": 1.0}}
        config = ReportConfig()

        assert "<td>7</td>" in self.viz_engine.generate_field_performance_table(field_metrics, config)
        assert '<div class="field-label">7</div>' in self.viz_engine.generate_field_performance_chart(field_metrics, config)

    def test_generate_field_performance_table_with_fallback_keys(self):
        """Test field performance table with fallback metric keys."""
        field_metrics = {
            "category": {
                "precision": 0.85,  # No 'cm_' prefix
                "recall": 0.78,
                "f1": 0.81,
                "tp": 20,
                "fd": 2,
                "fa": 3,
                "fn": 4
            }
        }
        
        config = ReportConfig()
        result = self.viz_engine.generate_field_performance_table(field_metrics, config)
        
        assert 'category' in result
        assert '0.850' in result
        assert '0.780' in result
        assert '0.810' in result
        assert '20' in result
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_confusion_matrix_colors')
    def test_generate_confusion_matrix_heatmap(self, mock_color_utils):
        """Test confusion matrix heatmap generation."""
        mock_color_utils.return_value = {
            'tp': '#28a745',
            'tn': '#17a2b8',
            'fd': '#ffc107',
            'fa': '#fd7e14',
            'fn': '#dc3545'
        }
        
        cm_data = {
            'tp': 45,
            'tn': 30,
            'fd': 5,
            'fa': 3,
            'fn': 7
        }
        
        result = self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        assert '<div class="cm-grid">' in result
        assert 'TP' in result
        assert 'TN' in result
        assert 'FD' in result
        assert 'FA' in result
        assert 'FN' in result
        assert '45' in result
        assert '30' in result
        assert '50.0%' in result  # 45/90 * 100
        assert '33.3%' in result  # 30/90 * 100
        assert 'border-left-color: #28a745' in result
        mock_color_utils.assert_called_once()
    
    def test_generate_confusion_matrix_heatmap_empty_data(self):
        """Test confusion matrix heatmap with empty data."""
        cm_data = {}
        
        result = self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        assert '<p>No confusion matrix data to visualize.</p>' in result
    
    def test_generate_confusion_matrix_heatmap_zero_total(self):
        """Test confusion matrix heatmap with all zero values."""
        cm_data = {
            'tp': 0,
            'tn': 0,
            'fd': 0,
            'fa': 0,
            'fn': 0
        }
        
        result = self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        assert '<p>No confusion matrix data to visualize.</p>' in result
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_confusion_matrix_colors')
    def test_generate_confusion_matrix_heatmap_missing_metrics(self, mock_color_utils):
        """Test confusion matrix heatmap with missing metrics."""
        mock_color_utils.return_value = {
            'tp': '#28a745',
            'tn': '#17a2b8',
            'fd': '#ffc107',
            'fa': '#fd7e14',
            'fn': '#dc3545'
        }
        
        cm_data = {
            'tp': 20,
            'tn': 15
            # Missing fd, fa, fn
        }
        
        result = self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        assert '<div class="cm-grid">' in result
        assert 'TP' in result
        assert 'TN' in result
        assert 'FD' in result  # Should still appear with 0 value
        assert '20' in result
        assert '15' in result
        assert '0' in result  # Missing values should default to 0
        assert '57.1%' in result  # 20/35 * 100
        assert '42.9%' in result  # 15/35 * 100
        assert '0.0%' in result   # 0/35 * 100


class TestFieldPerformanceValuesAreEscaped:
    """The confusion counts in the field performance table are rendered, not trusted.

    `generate_report` publicly accepts a plain dict, so a caller who assembles
    their own metrics — rather than passing a `ProcessEvaluation`, where these
    are library-computed ints — can put a string in `tp`/`fd`/`fa`/`fn`. Those
    four cells carried no escaping and no format spec, unlike the
    `precision`/`recall`/`f1` cells beside them, whose `:.3f` rejects a string
    by accident. #431 closed the `<script>` blob; this closes the same class in
    the table two functions over.
    """

    HOSTILE = {
        "ven<dor>&": {
            "cm_precision": 1.0,
            "cm_recall": 1.0,
            "cm_f1": 1.0,
            "tp": "<img src=x onerror=alert(1)>",
            "fd": "</td></tr></table><script>alert(2)</script>",
            "fa": "<img src=x onerror=alert(3)>",
            "fn": "</td></tr></table><script>alert(4)</script>",
        }
    }

    def _table(self, field_metrics):
        from stickler.reporting.html.report_config import ReportConfig
        from stickler.reporting.html.visualization_engine import VisualizationEngine

        return VisualizationEngine().generate_field_performance_table(
            field_metrics, ReportConfig()
        )

    @staticmethod
    def _parse(markup):
        """What a browser would actually see: elements, attributes, script bodies."""
        from html.parser import HTMLParser

        class Seen(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.attrs = []
                self.scripts = []
                self._in_script = False

            def handle_starttag(self, tag, attrs):
                self.attrs += [(tag, k, v) for k, v in attrs]
                if tag == "script":
                    self._in_script = True

            def handle_endtag(self, tag):
                if tag == "script":
                    self._in_script = False

            def handle_data(self, data):
                if self._in_script and data.strip():
                    self.scripts.append(data.strip())

        p = Seen()
        p.feed(markup)
        return p

    def test_a_hostile_value_is_inert_in_the_table(self):
        out = self._table(self.HOSTILE)
        seen = self._parse(out)
        # The characters survive as text; what must not survive is any parse of
        # them as markup.
        assert [a for a in seen.attrs if a[1].startswith("on")] == []
        assert seen.scripts == []
        assert "&lt;img src=x" in out
        assert "&lt;script&gt;" in out
        # the row key is a field name, which is its own injection route (#422)
        assert "ven<dor>&" not in out
        assert "ven&lt;dor&gt;&amp;" in out

    def test_the_confusion_matrix_cell_is_inert(self):
        """A numeric subclass with a hostile __str__ reaches the cell.

        `generate_confusion_matrix_heatmap` sums the counts first, so a plain
        string raises `TypeError` before rendering — which is why this looked
        unreachable. An `int` subclass sums and divides fine and then renders
        through `str()`, so the escape there is load-bearing, and this test
        reddens when it is reverted.
        """
        from stickler.reporting.html.report_config import ReportConfig
        from stickler.reporting.html.visualization_engine import VisualizationEngine

        class Hostile(int):
            def __str__(self):
                return "<img src=x onerror=alert(1)>"

        out = VisualizationEngine().generate_confusion_matrix_heatmap(
            {"tp": Hostile(1), "fd": 0, "fa": 0, "fn": 0}, ReportConfig()
        )
        seen = self._parse(out)
        assert [a for a in seen.attrs if a[1].startswith("on")] == []
        assert seen.scripts == []
        assert "<img src=x" not in out
        assert "&lt;img src=x" in out

    def test_ordinary_counts_still_render_unchanged(self):
        out = self._table(
            {
                "vendor": {
                    "cm_precision": 1.0,
                    "cm_recall": 1.0,
                    "cm_f1": 1.0,
                    "tp": 7,
                    "fd": 0,
                    "fa": 1,
                    "fn": 2,
                }
            }
        )
        for n in ("7", "0", "1", "2"):
            assert f"<td>{n}</td>" in out


class TestFormatSpecCellsAreEscaped:
    """``:.3f`` / ``:.1f`` cells must escape after formatting (#453).

    A numeric subclass whose ``__format__`` returns markup reaches the page
    unless the formatted string is escaped. Chart/table cells need only
    ``__format__``; the heatmap percentage needs ``__truediv__`` + ``__mul__``
    + ``__format__`` because the template formats ``(value / total) * 100``.
    """

    class HF(float):
        def __format__(self, spec):
            return "<img src=x onerror=alert(1)>"

    class HeatHostile(float):
        def __truediv__(self, other):
            return TestFormatSpecCellsAreEscaped.HeatCarrier(float(self) / float(other))

        def __rtruediv__(self, other):
            return TestFormatSpecCellsAreEscaped.HeatCarrier(float(other) / float(self))

    class HeatCarrier(float):
        def __mul__(self, other):
            return TestFormatSpecCellsAreEscaped.HeatHostileResult(float(self) * float(other))

        def __rmul__(self, other):
            return TestFormatSpecCellsAreEscaped.HeatHostileResult(float(other) * float(self))

    class HeatHostileResult(float):
        def __format__(self, spec):
            return "<img src=x onerror=alert(1)>"

    def _engine(self):
        from stickler.reporting.html.report_config import ReportConfig
        from stickler.reporting.html.visualization_engine import VisualizationEngine
        return VisualizationEngine(), ReportConfig()

    def test_table_precision_recall_f1_escape_format_markup(self):
        eng, config = self._engine()
        out = eng.generate_field_performance_table(
            {
                "vendor": {
                    "precision": self.HF(0.9),
                    "recall": self.HF(0.8),
                    "f1": self.HF(0.85),
                    "tp": 1,
                    "fd": 0,
                    "fa": 0,
                    "fn": 0,
                }
            },
            config,
        )
        assert "<img src=x" not in out
        assert "&lt;img src=x" in out

    def test_chart_bar_value_escapes_format_markup(self):
        eng, config = self._engine()
        out = eng.generate_field_performance_chart(
            {"vendor": {"f1": self.HF(0.75)}},
            config,
        )
        assert "<img src=x" not in out
        assert "&lt;img src=x" in out

    def test_heatmap_percentage_escapes_format_markup(self):
        eng, config = self._engine()
        out = eng.generate_confusion_matrix_heatmap(
            {"tp": self.HeatHostile(1), "tn": 0, "fd": 0, "fa": 0, "fn": 0},
            config,
        )
        assert "<img src=x" not in out
        assert "&lt;img src=x" in out

    def test_ordinary_format_values_still_render(self):
        eng, config = self._engine()
        table = eng.generate_field_performance_table(
            {
                "vendor": {
                    "precision": 0.9123,
                    "recall": 0.8,
                    "f1": 0.85,
                    "tp": 1,
                    "fd": 0,
                    "fa": 0,
                    "fn": 0,
                }
            },
            config,
        )
        chart = eng.generate_field_performance_chart(
            {"vendor": {"f1": 0.75}},
            config,
        )
        heat = eng.generate_confusion_matrix_heatmap(
            {"tp": 1, "tn": 0, "fd": 0, "fa": 0, "fn": 0},
            config,
        )
        assert "0.912" in table
        assert "0.750" in chart
        assert "100.0%" in heat
