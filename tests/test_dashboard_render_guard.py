from pathlib import Path
import unittest


class DashboardRenderGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (Path(__file__).resolve().parents[1] / "dashboard.html").read_text(
            encoding="utf-8"
        )
        start = cls.html.index("function render(){")
        end = cls.html.index("\n}\nlet strategyChart", start)
        cls.render = cls.html[start:end]

    def test_model_state_is_defined_inside_render_scope(self):
        ai_metrics = self.render.index('$("aiMetrics")')
        self.assertIn('const modelMeta=', self.render[:ai_metrics])
        self.assertIn('const modelReady=', self.render[:ai_metrics])

    def test_independent_sections_are_rendered(self):
        self.assertIn("renderIndependentPages();", self.render)

    def test_no_literal_newline_escape_between_dashboard_pages(self):
        self.assertNotIn(r"\\n\\n<section", self.html)


if __name__ == "__main__":
    unittest.main()
