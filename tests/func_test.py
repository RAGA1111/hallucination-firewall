# test_decomposer.py

from core.decomposer import decompose_claims

text = """
Apple was founded in 1976.
Steve Jobs co-founded Apple.
"""

result = decompose_claims(text)
