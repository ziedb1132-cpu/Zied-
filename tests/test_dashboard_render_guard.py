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

    def test_model_state_is_defined_inside_ai_learning_panel_scope(self):
        start = self.html.index("function renderAiLearningPanel(){")
        end = self.html.index("\n}", start)
        ai_panel = self.html[start:end]
        ai_metrics = ai_panel.index('$("aiMetrics")')
        self.assertIn('const modelMeta=', ai_panel[:ai_metrics])
        self.assertIn('const modelReady=', ai_panel[:ai_metrics])
        self.assertIn("renderAiLearningPanel();", self.render)

    def test_independent_sections_are_rendered(self):
        self.assertIn("renderIndependentPages();", self.render)

    def test_ai_learning_panel_renders_after_connection_even_if_general_render_fails(self):
        self.assertIn("function renderAiLearningPanel()", self.html)
        self.assertIn("try{renderAiLearningPanel();}catch(aiError)", self.html)
        self.assertIn("renderAiLearningPanel();", self.render)

    def test_ai_page_explains_learning_threshold_and_validation_separately(self):
        self.assertIn("Progression apprentissage", self.html)
        self.assertIn("Validation indépendante", self.html)
        self.assertIn("aiLearningNotice", self.html)

    def test_independent_analytics_render_after_general_render_failure(self):
        self.assertIn("try{renderIndependentPages();}catch(pageError)", self.html)

    def test_no_literal_newline_escape_between_dashboard_pages(self):
        self.assertNotIn(r"\n\n<section", self.html)


if __name__ == "__main__":
    unittest.main()
