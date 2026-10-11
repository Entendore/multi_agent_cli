# tools.py
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluates math expressions."""
    try: return str(eval(expression))
    except Exception as e: return f"Error: {e}"

@tool
def python_repl(code: str) -> str:
    """Execute Python code."""
    from langchain_experimental.utilities import PythonREPL
    try: return PythonREPL().run(code)
    except Exception as e: return f"Error: {e}"

@tool
def echo(text: str) -> str:
    """Echo text."""
    return text

@tool
def score_solution(solution: str, criteria: str = "clarity, correctness") -> str:
    """Score a solution 0-10."""
    s = 0.0
    if len(solution) > 50: s += 2
    if len(solution) > 150: s += 2
    if any(kw in solution.lower() for kw in ["step", "because", "therefore"]): s += 3
    if any(kw in solution.lower() for kw in criteria.lower().split(", ")): s += 3
    return f"{min(s, 10):.1f}/10"

@tool
def extract_hypotheses(text: str) -> str:
    """Extract hypotheses from text."""
    import re
    m = re.findall(r'H\d+\s*[:.]?\s*(.+)', text, re.IGNORECASE)
    if not m: m = re.findall(r'\d+\.\s*(.+)', text)
    return "\n".join(f"H{i+1}: {x.strip()}" for i, x in enumerate(m[:5])) if m else "None found"

@tool
def format_report(sections: str) -> str:
    """Format into report with headers."""
    lines = sections.strip().split("\n"); out = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if any(line.upper().startswith(kw) for kw in ["ABSTRACT", "INTRO", "METHOD", "RESULT", "DISCUSS", "CONCLUSION", "HYPOTHESIS"]): out.append(f"\n## {line}\n")
        else: out.append(line)
    return "\n".join(out)

def get_builtin_tools():
    return [calculator, python_repl, echo, score_solution, extract_hypotheses, format_report]
