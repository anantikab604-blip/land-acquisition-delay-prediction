"""
Step 6 (land acquisition project): write requirements.txt.

Uses the versions installed in the CURRENT Python environment, so make sure your
prompt shows (python-env) before you run it:
    python make_requirements.py
"""
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["pandas", "numpy", "scikit-learn", "joblib", "matplotlib", "streamlit"]

lines = []
for name in PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"Warning: {name} is not installed in this environment, skipped.")

with open("requirements.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Saved requirements.txt:")
print("\n".join(lines))
