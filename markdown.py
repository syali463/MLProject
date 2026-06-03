from markitdown import MarkItDown
import os 

current_path = os.getcwd()

file_path = os.path.join(current_path, "markdown.md")

md = MarkItDown()
result = md.convert("XGBoost_Crop_Yield_Validity_Analysis.pdf")

with open("markdown.md", "w", encoding="utf-8") as f:
    f.write(result.markdown)

print(result)
