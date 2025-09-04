from rapidfuzz import fuzz

print(fuzz.ratio("this is a test", "this is a test!"))  # 96