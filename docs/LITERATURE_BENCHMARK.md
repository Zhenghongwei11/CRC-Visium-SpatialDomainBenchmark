# Literature benchmark memo

## Purpose
Benchmark writing structure and evidence presentation against recent Q1/Q2 papers before final drafting.

## Target papers (Q1/Q2 benchmark set)
| paper_id | journal | year | identifier | domain relevance | writing tactics extracted |
|---|---:|---:|---|---|---|
| GB2024_ST_Benchmark | Genome Biology (Q1) | 2024 | https://doi.org/10.1186/s13059-024-03361-0 | Benchmarking clustering/alignment/integration in spatial transcriptomics; multi-dataset + multi-metric + runtime | Multi-metric results framed as "no single winner"; explicit scalability section; recommendation table; failures/NA are shown (not hidden) |
| NM2024_SpatialClust_Benchmark | Nature Methods (Q1) | 2024 | https://www.nature.com/articles/s41592-024-02215-8 | Benchmarking spatial clustering methods; relevant for how to discuss "domain calls" without ground truth | Use multiple metrics; emphasize robustness and continuity; avoid overclaiming method superiority beyond evaluated settings |
| NM2024_sST_Tech_Benchmark | Nature Methods (Q1) | 2024 | https://www.nature.com/articles/s41592-024-02325-3 | Technology benchmarking paper; useful for how to describe benchmarking datasets, fairness, and scope-limited recommendations | Early framing: define the gap, then define reference tissues + standard pipeline; explicit "serves multiple purposes" list; strong limitations about simulation/ground truth |

## Optional (open-access) tone reference
- PLOS Computational Biology methods paper (writing tone reference only): https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011935

## Q1 anchor-paper deep read
Anchor paper: GB2024_ST_Benchmark (Genome Biology 2024; open access)

### Results paragraph pattern (observed)
1. Start with the benchmark task and what was compared (methods + datasets + metrics).
2. Present a clear "what worked where" summary rather than a single winner claim.
3. Immediately follow with a constraint: tissue/platform specificity, dataset complexity, and metric trade-offs.
4. Add a runtime/scalability paragraph as its own subsection (including failures/NA).

### Discussion limitation phrasing pattern (observed)
- Limitations are specific and constructive (what is not supported and why), followed by "what would be needed next" (additional datasets, orthogonal validation, or protocol variants), without apologetic disclaimers.
- The paper avoids "universal best" language and instead uses "overall ranking under the evaluated settings" language.

### Figure-caption evidence style (observed)
- Captions explicitly tie panels to datasets/method groups and clarify what each metric measures.
- Failures are visually encoded (empty columns/NA) and explained in the caption or adjacent text.

## Revision checklist
- [ ] Introduction states (i) why domain calls are unstable in practice, (ii) why prespecified gates are needed, (iii) what this benchmark contributes beyond "yet another method comparison"
- [ ] Results paragraphs follow question -> finding -> evidence (Figure/Table/Source Data labels) -> limitation/scope -> transition
- [ ] No "best method" claims; all comparative statements explicitly scoped to datasets + preprocessing + K-grid + seed scheme
- [ ] Runtime/feasibility section is explicitly labeled scope-limited and shows failures/NA transparently
