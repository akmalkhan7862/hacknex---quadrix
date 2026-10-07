"""
Phase B Unit Tests — Query Analyzer

Covers:
  - analyze_query() backward-compatibility (all original keys present)
  - Intent detection: comparison, calculation, visual_analysis, fact_lookup, aggregation
  - Modality detection: text, table, chart, image
  - Period extraction: Q1-Q4, FY, years, months
  - Entity extraction
  - Cross-document flag
  - build_plan() -> Plan (all PHASE O scenario questions)
"""

import pytest
from backend.app.query.analyzer import analyze_query, build_plan
from backend.app.query.schemas import Plan, RetrievalStrategy


# ────────────────────────────────────────────
# Backward-compatibility: original dict keys
# ────────────────────────────────────────────

class TestAnalyzeQueryBackwardCompatibility:
    def test_returns_dict(self):
        result = analyze_query("What is revenue?")
        assert isinstance(result, dict)

    def test_all_original_keys_present(self):
        result = analyze_query("What was total revenue in Q4?")
        for key in ("question", "needs_text", "needs_table",
                    "needs_calculation", "keywords", "entities"):
            assert key in result, f"Missing key: {key}"

    def test_needs_text_always_true(self):
        result = analyze_query("anything at all")
        assert result["needs_text"] is True

    def test_needs_table_detected(self):
        result = analyze_query("What was production efficiency in Q4?")
        assert result["needs_table"] is True

    def test_needs_table_false_for_visual(self):
        result = analyze_query("What does the chart show?")
        # chart query does NOT set needs_table unless table keywords appear
        assert result["needs_table"] is False

    def test_needs_calculation_detected(self):
        result = analyze_query("Calculate the percentage increase.")
        assert result["needs_calculation"] is True

    def test_needs_calculation_false(self):
        result = analyze_query("Summarize the executive summary.")
        assert result["needs_calculation"] is False

    def test_keywords_not_empty_for_real_question(self):
        result = analyze_query("What exact value is reported for Q4?")
        assert len(result["keywords"]) > 0

    def test_stop_words_excluded_from_keywords(self):
        result = analyze_query("What is the value from the table?")
        kws_lower = [k.lower() for k in result["keywords"]]
        for sw in ("what", "the", "from", "is"):
            assert sw not in kws_lower

    def test_entities_extracted(self):
        result = analyze_query("Compare the Annual Report and Operations Report.")
        assert len(result["entities"]) >= 1


# ────────────────────────────────────────────
# Extended fields
# ────────────────────────────────────────────

class TestAnalyzeQueryExtendedFields:
    def test_new_fields_present(self):
        result = analyze_query("What is the trend in the production chart?")
        for key in ("intent", "query_type", "periods", "modalities",
                    "needs_visual", "cross_document"):
            assert key in result, f"Missing extended key: {key}"

    def test_needs_visual_chart(self):
        result = analyze_query("What trend does the production efficiency chart show?")
        assert result["needs_visual"] is True

    def test_needs_visual_false(self):
        result = analyze_query("What was production efficiency in Q4?")
        assert result["needs_visual"] is False

    def test_cross_document_flag(self):
        result = analyze_query("Compare the annual report and operations report.")
        assert result["cross_document"] is True

    def test_cross_document_false(self):
        result = analyze_query("What was the Q4 production efficiency?")
        assert result["cross_document"] is False


# ────────────────────────────────────────────
# Period extraction
# ────────────────────────────────────────────

class TestPeriodExtraction:
    def test_single_quarter(self):
        result = analyze_query("What was revenue in Q3?")
        assert "Q3" in result["periods"]

    def test_two_quarters(self):
        result = analyze_query("Compare Q2 and Q4 production efficiency.")
        assert "Q2" in result["periods"]
        assert "Q4" in result["periods"]

    def test_fiscal_year(self):
        result = analyze_query("What were FY2023 results?")
        assert any("FY" in p or "2023" in p for p in result["periods"])

    def test_calendar_year(self):
        result = analyze_query("Show revenue for 2022.")
        assert "2022" in result["periods"]

    def test_no_period(self):
        result = analyze_query("What is the company mission?")
        assert result["periods"] == []


# ────────────────────────────────────────────
# Modality detection
# ────────────────────────────────────────────

class TestModalityDetection:
    def test_text_always_present(self):
        result = analyze_query("Describe the executive summary.")
        assert "text" in result["modalities"]

    def test_table_detected(self):
        result = analyze_query("What was production efficiency in Q4?")
        assert "table" in result["modalities"]

    def test_chart_detected(self):
        result = analyze_query("What trend does the production efficiency chart show?")
        assert "chart" in result["modalities"]

    def test_image_detected(self):
        result = analyze_query("Show me the product image.")
        assert "image" in result["modalities"]

    def test_no_duplicates_in_modalities(self):
        result = analyze_query("Show the revenue table and the revenue chart.")
        assert len(result["modalities"]) == len(set(result["modalities"]))


# ────────────────────────────────────────────
# Intent detection
# ────────────────────────────────────────────

class TestIntentDetection:
    def test_comparison_intent(self):
        result = analyze_query("Compare Q2 and Q4 production efficiency.")
        assert result["intent"] == "comparison"
        assert result["query_type"] == "comparative"

    def test_calculation_intent(self):
        result = analyze_query("Calculate the percentage increase from Q2 to Q4.")
        assert result["intent"] == "calculation"

    def test_visual_intent(self):
        result = analyze_query("What trend is shown in the production efficiency chart?")
        assert result["intent"] == "visual_analysis"
        assert result["query_type"] == "visual"

    def test_fact_lookup_intent(self):
        result = analyze_query("What was production efficiency in Q4?")
        assert result["intent"] == "fact_lookup"
        assert result["query_type"] == "factual"

    def test_cross_doc_intent(self):
        result = analyze_query("Compare the annual report and operations report.")
        assert result["cross_document"] is True


# ────────────────────────────────────────────
# build_plan() returning Plan
# ────────────────────────────────────────────

class TestBuildPlan:
    def test_returns_plan_instance(self):
        plan = build_plan("What was production efficiency in Q4?")
        assert isinstance(plan, Plan)

    def test_plan_question_preserved(self):
        q = "What was production efficiency in Q4?"
        plan = build_plan(q)
        assert plan.question == q

    def test_plan_q1_fact_lookup(self):
        """PHASE O Q1: fact lookup → table retrieval."""
        plan = build_plan("What was production efficiency in Q4?")
        assert plan.intent == "fact_lookup"
        assert "table" in plan.modalities
        assert "Q4" in plan.periods

    def test_plan_q2_exact_value(self):
        """PHASE O Q2: exact value → BM25 + table."""
        plan = build_plan("What exact value is reported for Q4?")
        assert "Q4" in plan.periods
        assert "text" in plan.modalities   # at minimum text

    def test_plan_q3_trend_chart(self):
        """PHASE O Q3: trend chart → visual retrieval."""
        plan = build_plan("What trend is shown in the production efficiency chart?")
        assert plan.intent == "visual_analysis"
        assert "chart" in plan.modalities
        assert plan.query_type == "visual"

    def test_plan_q4_comparison(self):
        """PHASE O Q4: compare Q2 and Q4 → hybrid + table + visual."""
        plan = build_plan("Compare Q2 and Q4 production efficiency.")
        assert plan.intent == "comparison"
        assert "Q2" in plan.periods
        assert "Q4" in plan.periods
        assert "table" in plan.modalities

    def test_plan_q5_cross_doc(self):
        """PHASE O Q5: cross-document → cross_doc flag set."""
        plan = build_plan("Compare the annual report and operations report.")
        assert plan.cross_document is True
        assert plan.intent == "cross_doc_synthesis" or plan.intent == "comparison"

    def test_plan_q6_calculation(self):
        """PHASE O Q6: calculate percentage → retrieve evidence only (no calc)."""
        plan = build_plan("Calculate the percentage increase.")
        assert plan.needs_calculation is True
        assert plan.intent == "calculation"

    def test_plan_entities_populated(self):
        plan = build_plan("Compare the Annual Report and Operations Report.")
        assert len(plan.entities) >= 1

    def test_plan_filters_empty_by_default(self):
        plan = build_plan("What is total revenue?")
        assert plan.filters == {}

    def test_plan_sub_questions_empty_by_default(self):
        plan = build_plan("What is total revenue?")
        assert plan.sub_questions == []
