markdown# ECLIPSE Hash Verification

This directory contains tools for independent verification of ECLIPSE v4.1 data split integrity.

## Quick Verification
```bash
python verify_eclipse_hashes.py
What This Verifies

✅ Train/holdout split has not been modified
✅ Sacred seed (2025) was used correctly
✅ Hashes match those published in the paper
✅ Timestamp authenticity

Expected Output
✅ VERIFICATION SUCCESSFUL
The data split is cryptographically verified.
Hashes match those published in the paper.
Published Hashes (from paper)
Train indices (n=107):
6afe17de49ff87cd5493674619b4ae9d827c40e9aa362bc934e0043b64f3dd45
Holdout indices (n=46):
8ed633226b92de6dc5722f7c6cb8d4176fe940d889bcaa4d2610817c732b177c
Paper Reference
Sjöberg Tala, C. A. (2025). ECLIPSE: A Systematic Falsification Framework for Consciousness Science.
DOI: 10.5281/zenodo.16747014
Contact
For questions about hash verification:

Email: cst@afhmodel.org
GitHub: github.com/camilosjobergtala/AFH-MODEL


---

### **3. `expected_hashes.txt`**
ECLIPSE v4.1 - Expected SHA-256 Hashes
Train indices (n=107):
6afe17de49ff87cd5493674619b4ae9d827c40e9aa362bc934e0043b64f3dd45
Holdout indices (n=46):
8ed633226b92de6dc5722f7c6cb8d4176fe940d889bcaa4d2610817c732b177c
Sacred Seed: 2025
Timestamp: 2025-07-22T22:19:46.975849
These hashes are published in:
Sjöberg Tala, C. A. (2025). ECLIPSE: A Systematic Falsification Framework.
DOI: 10.5281/zenodo.16747014