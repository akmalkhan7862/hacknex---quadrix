"""
Calculator / Python Reasoning Module for MDI
Provides calculation extension point for formula and numerical verification.
"""

import re
from typing import Dict, Any, List

class PythonCalculatorExtension:
    """Extension point for Phase 2/3 Python reasoning engine."""
    
    def evaluate_expression(self, expr: str) -> Dict[str, Any]:
        """Safely evaluate simple arithmetic expressions."""
        try:
            # Simple math regex check (numbers and basic operators)
            clean_expr = re.sub(r"[^\d\.\+\-\*\/\(\)\s]", "", expr)
            if not clean_expr.strip():
                return {"success": False, "error": "Invalid math expression"}
            result = eval(clean_expr)
            return {"success": True, "expression": clean_expr, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

global_calculator = PythonCalculatorExtension()
