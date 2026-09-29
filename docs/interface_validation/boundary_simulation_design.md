# Known-boundary calibration design

## Scope

This revision-stage simulation measures the validity of a regional estimator.
It does not fit BayesSpace or neural networks, simulate a validated CRC barrier
signature, or demonstrate an algorithm-specific mechanism. The historical
anchor, region-selection and block-bootstrap functions are reused unchanged.

The manifest defines 48 core cases: two section sizes, two mixing levels,
two signal widths, three signal amplitudes including zero, and two noise
correlation lengths. Four additional cases vary graph degree (4 or 8) and
one-step majority smoothing at a fixed declared core condition. Topology
cases form a separate sensitivity experiment, not a full interaction grid.
Each case has 100 independent realizations; both estimators and three maps
share the same realization.

## Geometry and signal

Centers occupy a staggered hexagonal lattice with 100 micrometre pitch and
55 micrometre nominal spot diameter. A known sinusoidal interface separates
tumor-center and stromal-center labels. These idealized center labels are
not simulated pathology annotations. Distances are measured to tumor spot
centers, and connectivity uses the historical array-coordinate neighbor
rule; a physical hexagonal neighbor rule is not silently substituted.

Latent stromal means decay exponentially away from the interface and vary
sinusoidally along it. Width is a signal length scale, not a cytokine model.
Observed means mix stromal and fixed tumor means using a distance-dependent
fraction capped by `mixing_max`. This is controlled admixture, not a cell
count or exact histological integration over the disc. Nominal spot diameter
provides context and is not a varied generator parameter.

Noise combines an independent Gaussian nugget with a spatially filtered
Gaussian field. Variance is normalized using known filter weights, not the
variance of each observed realization. Each spot has the declared marginal
SD, including at edges. Spatial correlation changes without changing that SD.

## Three truths

For each anchor or retained group, the population median is the root of its
equal-weight Gaussian mixture CDF over spot locations. It is not a noisy
sample median or the median of heterogeneous means.

1. Latent anchor truth describes the unmixed regional signal.
2. Measured anchor truth describes the mixed-spot signal at the same locations.
3. Map-selected truth describes the mixed-spot signal at retained locations.

Report admixture distortion (2 minus 1), selection distortion (3 minus 2),
and estimation error (estimate minus 3) separately. Paired truth is the
difference between population contrasts. Correlation affects sampling
variation but does not change the marginal mixture CDF defining the target.

## Maps and negative controls

The identity map reproduces tissue compartments. The reference map adds
a sinusoidally varying outward offset of at most 75 micrometres. The
alternative adds a further 50 micrometre oscillation. These geometric
perturbations do not use outcome values and do not emulate the seven study
algorithms. Majority smoothing applies only to shifted maps. Record ARI,
near/far retention, overlap, and eligibility for every map.

Zero-amplitude cases test false signal/reversal. Identical memberships must
produce exactly zero paired change and [0, 0] intervals. Degenerate controls
remain visible but are excluded from nondegenerate paired-coverage gates.

## Inference

The historical implementation divides the entire array-coordinate extent
into a 6-by-6 grid; it does not define blocks of 6-by-6 spots. Reference and
alternative blocks are resampled jointly within each distance band. Record
attempted, accepted, and used draws to evaluate conditional rejection.

Summaries by case/map/estimator include bias, RMSE, 95% coverage, true-null
interval rejection, legacy support loss, supported reversal, and paired
calibration. Binary rates have Wilson intervals over independent sections.
Screening requires 100 replicates, at least 80% evaluability, absolute bias
at most 0.10 noise SD, and coverage Wilson lower limit at least 0.90. Under
true zero, the false-rejection Wilson upper limit must be at most 0.10.
These are calibration tolerances, not biological margins or proof of exact
95% coverage. Per-condition uncertainty must be reported; pooled coverage
cannot conceal failures. No familywise biological claim is made.

Support and paired-change thresholds use the complete fixed manifest sweep.
False loss means an estimated supported anchor followed by unsupported map
contrast when both known population contrasts exceed the same observed
MAD-based margin in the same direction. Export this conditional denominator;
do not label true selection distortion a classification error. True zero
uses the fixed numeric tolerance. Threshold rates are secondary descriptors.

## Freeze and resource boundary

The `freeze` command binds UTC time and SHA-256 for the manifest, design,
protocol, simulation script and downstream helper. Runs reject mismatches.
Smoke uses four fixed cases and two replicate indices, totaling eight
simulated sections. It checks implementation and is not claim evidence.
Full outputs are isolated from smoke outputs.

Use one CPU worker and stop between replications at a 300-second invocation
ceiling, with checksum-verified resume. No model training, data download or
Colab session is involved. A full run satisfies task 2.5 only after every
case/replicate is attempted and result hashes pass. Any later change to
the generator or estimator needs a versioned amendment with outcome
visibility stated; the old results are preserved.
