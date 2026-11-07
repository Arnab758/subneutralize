# SubNeutralize v0.2.0: Academic Paper LaTeX Package

**Title:** SubNeutralize: Latent Manifold Steering and Stateful Multi-Turn Eviction in Frontier Reasoning Models  
**Author:** Arnab Dutta (Founder & Lead Researcher, SubNeutralize)  
**Target:** NeurIPS / ICLR / arXiv Preprint  

---

## Files in this Package
1. **`main.tex`**: Complete two-column IEEE/NeurIPS LaTeX source code featuring formal mathematical theorems, definitions, proofs, the 4-arm experimental protocol, and comparative telemetry tables.
2. **`references.bib`**: BibTeX bibliography including DeepSeek-R1, OpenAI o1, Qwen2.5-Coder, SWE-bench, Aider, and mechanistic interpretability citations.

---

## 1-Click Upload to Overleaf
1. Zip the `paper/` directory:
   - Select `main.tex` and `references.bib`.
   - Compress into `subneutralize_paper.zip`.
2. Go to [Overleaf.com](https://www.overleaf.com).
3. Click **New Project** → **Upload Project**, and select `subneutralize_paper.zip`.
4. Overleaf will automatically compile the PDF manuscript.

---

## Local Compilation via PDFLaTeX
If you have a local TeX distribution (TeX Live, MiKTeX, MacTeX):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
