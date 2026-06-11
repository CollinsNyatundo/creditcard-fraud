# Raw Ideation Candidates — creditcard-fraud
_Session: 2026-06-09 | Mode: surprise-me | Frames: 6_

---

## Frame 1 — Pain & Friction
_What is consistently slow, broken, or annoying for users, operators, or the system?_

---

### 1.1 O(W) Rolling Aggregate Recalculation on Every Prediction

**title:** Rolling Aggregate Recomputation Tax

**summary:** Every call to `/predict` deserializes the entire Redis list for a card, then iterates through it to compute `amt_mean_3`, `amt_std_5`, and ~8 other rolling statistics from scratch (`predict.py` L140–196). At WINDOW_SIZE=10 this is acceptable, but under load (100 req/s) the cumulative deserialize-and-iterate cost is a constant tax added to every prediction. No incremental update path exists.

**basis:** direct: [`app/routes/predict.py` L140–196 — `card_history = get_card_history(card_id)` followed by `for tx in card_history[-window:]` loops for each aggregate]

**why_it_matters:** This is pure wasted CPU on every hot path. A Redis ZSET + running aggregate hash eliminates the iteration cost entirely, cutting feature-computation from O(W) to O(1).

**meeting_test:** How much of our p95 latency budget is consumed by rolling aggregate recomputation, and can we kill it with a single ZSET migration?

---

### 1.2 SHAP BackgroundTasks Starvation During Fraud Bursts

**title:** SHAP Event-Loop Starvation

**summary:** `BackgroundTasks` in FastAPI runs in the same event loop thread as the HTTP handler. When a fraud burst arrives (e.g., 20 flagged transactions in a second), 20 TreeSHAP computations queue up in the same event loop, blocking subsequent predictions from starting. This is the classic "background tasks that aren't really background" anti-pattern.

**basis:** direct: [`app/routes/predict.py` L410–418 — `background_tasks.add_task(compute_shap_explanation, ...)` for every `is_fraud=True` prediction]

**why_it_matters:** During the moment when fraud is most active — exactly when operators need fast response — the API slows to a crawl because SHAP is blocking the event loop.

**meeting_test:** Have we load-tested the endpoint during simulated fraud bursts, and what does p99 latency look like when 20% of requests are flagged?

---

### 1.3 MLflow Hard-Crash Dependency at Startup

**title:** MLflow Single Point of Failure

**summary:** The model is loaded by connecting to MLflow tracking server at startup (`main.py` L44). If MLflow is unreachable, the container crashes immediately with no fallback or grace period. Operators have no way to deploy a hotfix model or roll back without a running MLflow instance.

**basis:** direct: [`app/main.py` L44 — `model = mlflow.lightgbm.load_model(...)` inside lifespan with no try/catch]

**why_it_matters:** A monitoring tool (MLflow) has become a hard runtime dependency of a latency-critical service. These two concerns should be decoupled.

**meeting_test:** What is our recovery procedure when MLflow goes down during a deployment? Is there a fallback model path on disk?

---

### 1.4 Metrics Drift Across Five Separate Documents

**title:** Metrics Documentation Rot

**summary:** Model performance numbers (F1, AUC-PR, precision, recall) appear in at least 4 places: README.md, business_impact.md, evaluation JSON artifacts, and inline code comments. These have already drifted from each other. Any model update requires manually hunting and updating every location, which never happens in practice.

**basis:** direct: [Discrepancy between README.md performance table and `model_evaluation_results.json` values noted during codebase scan]

**why_it_matters:** Stale metrics in documentation mislead stakeholders about actual model performance and erode trust in the project's claims.

**meeting_test:** Which document has the ground truth metrics right now, and how many of us can answer that confidently?

---

### 1.5 `final_model_evaluation.py` Silent Runtime Crash

**title:** Evaluation Script Broken in Production

**summary:** `final_model_evaluation.py` calls `.predict_proba()` on a raw LightGBM `Booster` object, which doesn't have that method (it's a `sklearn.LGBMClassifier` method only). The script crashes at evaluation time, making the evaluation pipeline silently broken. No CI test catches this because integration tests don't run the evaluation script on a real model.

**basis:** direct: [`data/src/final_model_evaluation.py` — `model.predict_proba(X_test)` where model is loaded via `lgb.Booster()` not `lgb.LGBMClassifier`]

**why_it_matters:** We have no working automated evaluation path. Every time we want model metrics, an engineer must manually inspect and fix the script before running it.

**meeting_test:** Can anyone on the team run `python final_model_evaluation.py` right now and get a result?

---

### 1.6 Artifact Proliferation: 5+ Startup Files with No Versioning Contract

**title:** Unversioned Artifact Bundle Fragility

**summary:** The app loads `preprocessor.pkl`, `behavior_clusterer.pkl`, `feature_list.json`, `threshold_config.json`, and the MLflow model at startup — 5 separate artifacts with no shared version identifier. If any one was produced from a different training run than the others, the pipeline silently corrupts predictions. There is no manifest or hash check.

**basis:** direct: [`app/main.py` L44–67 — sequential independent file loads with no version compatibility check]

**why_it_matters:** As the model is retrained, the probability of a version mismatch between artifacts grows. Silent corruption is worse than a loud crash.

**meeting_test:** How do we guarantee that the preprocessor.pkl in production was trained with the same dataset as the current model?

---

### 1.7 `lightweight_feature_engineering.py` Overwrites Main Pipeline Data

**title:** Script Clobbering Shared Data Files

**summary:** Both `lightweight_feature_engineering.py` and `minimal_feature_engineering.py` write to the same `*_enhanced.csv` filenames used by the main pipeline. Running either script during active development silently replaces the dataset the main pipeline depends on, with no warning.

**basis:** direct: [Output filenames in `data/src/lightweight_feature_engineering.py` and `data/src/minimal_feature_engineering.py` match those consumed by `feature_engineering.py`]

**why_it_matters:** A developer running "the lightweight version to test quickly" can silently invalidate the main training run without realizing it.

**meeting_test:** Is there a namespace or output-path convention that prevents lightweight scripts from overwriting main pipeline data?

---

## Frame 2 — Inversion, Removal & Automation
_Invert a painful step, remove it entirely, or automate it away._

---

### 2.1 Invert: Push Aggregates, Don't Pull History

**title:** Pre-computed Aggregate Push on Write

**summary:** Instead of pulling the full card history on every read and computing rolling aggregates (the current O(W) pattern), invert the model: update aggregates incrementally each time a transaction is written to Redis. A single `HINCRBY` / `HINCRBYFLOAT` on an aggregate hash at write time means the read path just does `HMGET` — a single round-trip returning pre-computed values.

**basis:** external: [Feast feature store "write-time materialization" pattern; Redis HASH HINCRBYFLOAT for running sum/count]

**why_it_matters:** Every read becomes O(1). The aggregate computation work shifts from the hot read path (100 req/s × every prediction) to the write path (far lower frequency).

**meeting_test:** If we move aggregate computation to write-time, what is the new complexity of the prediction hot path?

---

### 2.2 Remove: Eliminate Python SHAP — Use Native `pred_contrib`

**title:** Kill the Python SHAP Dependency

**summary:** LightGBM's C++ engine computes SHAP values natively when `pred_contrib=True` is passed to `model.predict()`. The Python `shap` library wraps this same computation with significant overhead (Python object construction, array manipulation, compatibility layers). Switching to `pred_contrib=True` removes the `shap` library dependency entirely from the hot path.

**basis:** external: [LightGBM docs: `predict(data, pred_contrib=True)` returns SHAP values directly from C++ TreeSHAP; benchmarks show 1.5–3× speedup vs Python shap library]

**why_it_matters:** Removing the Python shap library from the prediction path reduces both latency and the risk of version incompatibilities with LightGBM upgrades.

**meeting_test:** Do we currently use any shap library features beyond TreeSHAP values (e.g., force plots, waterfall charts) that would block switching to pred_contrib?

---

### 2.3 Automate: CI Metrics Freeze — Auto-Update Docs on Model Artifact Change

**title:** CI-Driven Metrics Documentation Sync

**summary:** Add a CI step that extracts model metrics from the canonical JSON artifact after each training run and automatically rewrites the metrics tables in README.md and business_impact.md. The commit is created by the CI bot. No human manually updates documentation.

**basis:** reasoned: [Metrics drift exists because documentation update is a manual step that is consistently skipped. Automating the step removes the human failure mode entirely.]

**why_it_matters:** Eliminates an entire class of documentation rot with a one-time CI investment. All downstream consumers (stakeholders, auditors) always see current numbers.

**meeting_test:** What would it take to add a `make update-docs` step in our GitHub Actions pipeline triggered by a successful model training run?

---

### 2.4 Invert: Gate SHAP Behind a Probability Threshold, Not Behind `is_fraud`

**title:** Probability-Gated Explainability

**summary:** Currently SHAP runs for every transaction flagged as fraud (`is_fraud=True`). But the most operationally valuable explanations are for borderline cases near the decision threshold — high-confidence frauds don't need explanation (the score itself is sufficient). Invert the gate: compute SHAP only when `0.4 < probability < 0.85`, and skip it for clear-cut cases.

**basis:** external: [Gated explainability pattern from Stripe Radar and Kount fraud systems; "explanation budget" allocation based on uncertainty rather than outcome]

**why_it_matters:** Dramatically reduces SHAP computation volume while focusing explanatory resources exactly where human review adds the most value.

**meeting_test:** What fraction of our fraud flags today are high-confidence (>0.9 probability), and would skipping SHAP for those be operationally acceptable?

---

### 2.5 Remove: Eliminate Hardcoded Paths via Runtime Path Resolution

**title:** Automate Path Resolution with `pathlib.Path(__file__).parent`

**summary:** Multiple debug scripts hardcode `/app/realtime_credit_card_1507` as an absolute path, which works in containers but breaks on any developer's local machine. Replace all hardcoded paths with `pathlib.Path(__file__).parent.parent / "data"` or an environment variable with a sensible default. This is a one-hour fix that eliminates an entire class of "works in container, broken locally" bugs.

**basis:** direct: [Debug scripts referencing `/app/realtime_credit_card_1507`; `pathlib` is already used elsewhere in the codebase]

**why_it_matters:** Every new contributor wastes time debugging path errors before they can run anything locally. This is pure friction with zero upside.

**meeting_test:** How long does a new contributor currently take to get the pipeline running locally, and how much of that time is path debugging?

---

### 2.6 Automate: Artifact Version Manifest with Hash Pinning

**title:** Artifact Bundle Manifest Auto-Generated at Training Time

**summary:** At the end of each training run, automatically write a `model_manifest.json` containing: training run ID, SHA-256 hashes of all artifact files (preprocessor.pkl, clusterer.pkl, feature_list.json), and the git commit. The app validates this manifest at startup. If any hash mismatches, startup fails loudly instead of silently producing wrong predictions.

**basis:** reasoned: [Content-addressable artifact bundles are standard in ML deployment — used by BentoML, MLflow model signatures, and Seldon Core; currently absent from this codebase]

**why_it_matters:** Converts a silent corruption risk (mismatched artifact versions) into a loud, actionable startup failure.

**meeting_test:** How much would it cost to add manifest generation to the training pipeline and validation to the startup sequence?

---

### 2.7 Remove: Consolidate Lightweight/Minimal Scripts into Parameterized Pipeline

**title:** Eliminate Script Proliferation with a `--mode` Flag

**summary:** `feature_engineering.py`, `lightweight_feature_engineering.py`, and `minimal_feature_engineering.py` are three parallel scripts doing the same job with different scope. Replace all three with one script taking a `--mode full|lightweight|minimal` argument and writing to mode-namespaced output directories (e.g., `data/processed/full/`, `data/processed/minimal/`). This removes the clobbering problem structurally.

**basis:** direct: [Three separate scripts all writing to `*_enhanced.csv`; reasoned: parameterized pipelines with namespaced outputs is standard practice in data engineering]

**why_it_matters:** Eliminates the file-clobbering bug permanently and reduces codebase surface area by 2/3 for this module.

**meeting_test:** Are there meaningful semantic differences between the three scripts that a `--mode` flag couldn't capture, or are they just duplicates with different subset sizes?

---

## Frame 3 — Assumption-Breaking & Reframing
_What is treated as fixed that is actually a choice? Reframe one level up or sideways._

---

### 3.1 Assumption: The Model Needs All Features to Score

**title:** Feature-Sparse Fast Path for Low-Risk Transactions

**summary:** The current pipeline computes all ~30 engineered features for every transaction before scoring. But a simpler model (3–5 features: velocity, amount z-score, hour) can correctly classify 70%+ of transactions as clearly legitimate with high confidence. Reframe: use a cascade — a tiny model for easy cases, the full LightGBM only for uncertain ones.

**basis:** external: [Cascade classifier pattern from Viola-Jones face detection; "easy negatives" elimination in AdTech bidding pipelines; Microsoft's two-stage fraud scoring architecture]

**why_it_matters:** The majority of compute is spent scoring transactions that are obviously legitimate. A cascade can cut feature computation by 50–70% with no accuracy loss.

**meeting_test:** What does the distribution of fraud probabilities look like — how many transactions score below 0.05, where a simple rule would suffice?

---

### 3.2 Assumption: SHAP Is an Output for Humans

**title:** SHAP Values as Feedback Signal for Model Updates

**summary:** SHAP explanations are treated as a human-readable output artifact. Reframe: use aggregated SHAP values as a feature importance signal that feeds back into model retraining decisions. When SHAP indicates a feature's contribution has shifted significantly over time, trigger drift alerting. The explanation layer becomes an automated monitoring layer.

**basis:** external: [Evidently AI's SHAP-based drift detection; "explanation-driven monitoring" pattern described in Google's ML system design guide]

**why_it_matters:** We already pay the SHAP computation cost. Using those values for automated monitoring doubles the value at zero additional compute cost.

**meeting_test:** Do we currently use SHAP outputs for anything other than the explanation endpoint, and could we route them to Evidently for drift detection?

---

### 3.3 Assumption: Card History Is a List of Raw Amounts

**title:** Replace Raw History with Behavioral Sketch

**summary:** The current model stores raw transaction amounts in Redis history (`card_history` list) and then computes statistics from them. Reframe: store a Count-Min Sketch or HyperLogLog alongside a running EWMA of amounts. The sketch captures distributional behavior in a fixed 1KB structure regardless of history length, rather than growing linearly.

**basis:** external: [Count-Min Sketch for frequency estimation (Cormode & Muthukrishnan 2005); Redis HyperLogLog built-in; streaming statistics without raw data storage]

**why_it_matters:** History storage grows unboundedly; behavioral sketches are O(1) space and O(1) update. A card with 10 years of history costs the same as one with 10 transactions.

**meeting_test:** What is the maximum Redis memory we would commit to history storage under 1M active cards, and does sketching change that equation?

---

### 3.4 Assumption: Fraud Is a Binary Label

**title:** Continuous Fraud Risk Score as First-Class API Output

**summary:** The API returns `is_fraud: bool` alongside `fraud_probability: float`, but downstream consumers (dashboard, alerts) make their own threshold decisions. The binary label is a premature discretization that loses information. Reframe: remove `is_fraud` from the contract entirely and let consumers apply context-appropriate thresholds (stricter for high-value transactions, looser for microcharges).

**basis:** reasoned: [Decision theory: downstream agents with different loss functions should apply their own thresholds; the API is applying one threshold for all contexts]

**why_it_matters:** A travel purchase at 2AM looks different to a customer service agent versus an automated block system. Both should consume the same score and apply different thresholds.

**meeting_test:** Do all current consumers of `is_fraud` use the same threshold logic, or are some duplicating the threshold check?

---

### 3.5 Assumption: The Model Is Updated by Engineers

**title:** Merchant-Specific Micro-Tuning via Feedback Loop

**summary:** The model is a single global LightGBM trained on all card data. Reframe: allow high-volume merchants to provide labeled feedback (confirmed fraud / confirmed legitimate from chargebacks) that creates per-merchant model calibration layers (Platt scaling or isotonic regression on top of the global model). No retraining; just fast online calibration.

**basis:** external: [Platt scaling for classifier calibration; multi-task learning with shared base + entity-specific heads; Visa/Mastercard network-level model + issuer-specific override pattern]

**why_it_matters:** Different merchants have different fraud profiles. A gas station has different risk patterns than a luxury jeweler. Global calibration leaves significant precision on the table.

**meeting_test:** Do we have per-merchant feedback data from chargebacks that is currently unused, and how large is the largest merchant's transaction volume?

---

### 3.6 Assumption: Hour of Day Is a Raw Feature

**title:** Replace Hour with Behavioral Deviation from Cardholder Pattern

**summary:** The model uses `hour` as a raw feature (0–23), which means 2AM is inherently suspicious. But for a night-shift worker, 2AM is normal. Reframe: replace the raw hour feature with "deviation from this cardholder's historical hour distribution" — a z-score of the current hour relative to the card's own history. This turns an absolute feature into a relative behavioral signal.

**basis:** reasoned: [Population-level hour patterns are noisy; cardholder-relative deviation is a much stronger fraud signal; this is the same principle behind user behavior analytics in SIEM systems]

**why_it_matters:** This single feature substitution would likely improve model performance with no architectural changes — just a different column computed from existing Redis history data.

**meeting_test:** How much variance in fraud probability is currently explained by `hour` vs. by `hour_deviation_from_cardholder_baseline`, and can we A/B test this offline?

---

## Frame 4 — Leverage & Compounding
_Choices that, once made, make many future moves cheaper or stronger._

---

### 4.1 Redis Schema Migration to Versioned Hash Namespace

**title:** Versioned Feature Hash as Universal Leverage Point

**summary:** Migrating card history from a Redis List of raw amounts to a Hash of named features (using `HMSET card:{id}:v2 amt_mean_3 X amt_std_5 Y ...`) is a one-time investment that unlocks: O(1) feature reads, schema evolution without migration (just add new hash fields), multi-version coexistence for A/B testing new features, and atomic multi-field updates. Every future feature engineering change becomes a hash field addition, not a pipeline rewrite.

**basis:** external: [Feast entity-collocated Redis Hash materialization pattern; Redis HMGET batching for multi-feature retrieval in a single round-trip]

**why_it_matters:** This is the highest-leverage single change in the codebase. It makes every subsequent feature engineering improvement cheaper to ship and test.

**meeting_test:** What is the migration plan from List to Hash for existing card histories, and can we run both schemas in parallel during the cutover?

---

### 4.2 Canonical Metrics Registry as Single Source of Truth

**title:** Machine-Readable Metrics Registry

**summary:** Introduce a `metrics_registry.yaml` that is the single authoritative source for all model performance numbers. CI reads from this file to populate README badges, Grafana dashboards, and the business_impact.md table. The file is written by the training pipeline and committed as a build artifact. All other metric references are template substitutions from this registry.

**basis:** reasoned: [Single-source-of-truth principle; eliminates metrics drift by construction; Grafana datasource + README badge generation from a shared YAML is a common MLOps pattern]

**why_it_matters:** Once established, every future model update gets automatic documentation without human intervention. The pattern also makes metrics auditable via git history.

**meeting_test:** Could we implement `make update-metrics` in one sprint, and what CI trigger should write the registry file?

---

### 4.3 Artifact Bundle with Schema Version Field

**title:** Model Artifact Schema Version Contract

**summary:** Add a `schema_version` field to every serialized artifact (preprocessor.pkl metadata, feature_list.json, threshold_config.json). The app enforces that all loaded artifacts share the same schema version at startup. This is a 30-minute code change that prevents all future artifact mismatch bugs — which will otherwise occur on every model update for the lifetime of the project.

**basis:** reasoned: [Schema versioning is a standard pattern in serialization systems (Avro, Protobuf); applying it to ML artifacts catches mismatches before silent corruption occurs]

**why_it_matters:** The cost is trivial; the benefit compounds forever. Every future retrain, the team can be confident that the artifact bundle is internally consistent.

**meeting_test:** What is the simplest implementation of a schema version check that could be added to the startup sequence this sprint?

---

### 4.4 Structured Logging as the Analytics Foundation

**title:** Prediction Log as First-Class Event Stream

**summary:** Currently prediction results are stored as strings in Redis (`prediction_writer.py`). Converting to structured JSON logs (with card_id, amount, probability, features, timestamp, model_version) sent to a log aggregator (CloudWatch, Loki) creates a queryable analytics layer for free. This log stream becomes the foundation for drift monitoring, retrospective analysis, A/B test measurement, and compliance audit trails.

**basis:** direct: [`app/services/prediction_writer.py` — basic Redis write; reasoned: structured event logs are the standard foundation for ML observability in production systems]

**why_it_matters:** Every future capability — model comparison, drift detection, fraud pattern analysis — builds on this structured log. The sooner it's in place, the more historical data accumulates.

**meeting_test:** What format are prediction logs in today, and how long does it take to answer "what was the model's average confidence on transactions flagged last Tuesday"?

---

### 4.5 Test Fixture with Deterministic Model Snapshot

**title:** Golden Model Fixture for Integration Testing

**summary:** Train and commit a tiny (100-sample) deterministic LightGBM model and its full artifact bundle (preprocessor, clusterer, feature list, threshold config) to the test fixtures directory. Every integration test runs against this fixture, not against an MLflow-fetched production model. This single investment enables: fast CI, offline testing, deterministic results, and the ability to test prediction correctness.

**basis:** reasoned: [Test fixture models are standard in ML systems testing — used by Hugging Face, scikit-learn's own test suite, and BentoML; currently absent from this codebase which has no integration tests against a real model]

**why_it_matters:** All future integration tests become trivial to write. The 80% coverage gate in CI gains teeth — tests actually exercise the full prediction path.

**meeting_test:** How long would it take to train a deterministic 100-sample fixture model and commit it, and what tests would that immediately unlock?

---

### 4.6 FastAPI Dependency Injection for Model Loading

**title:** Dependency-Injectable Model Provider

**summary:** Refactor model loading from module-level globals into a FastAPI dependency (`Depends(get_model)`). This enables: swapping models per-request in tests without mocking, loading different model versions for A/B experiments, graceful reload without restart (hot swap), and clear ownership of model lifecycle. The current global state pattern makes all of these impossible.

**basis:** external: [FastAPI dependency injection docs; "service locator vs dependency injection" in clean architecture; used by all production FastAPI ML services at scale]

**why_it_matters:** This refactor unlocks model A/B testing, hot reload, and proper test isolation simultaneously. It's architectural debt that compounds in cost the longer it's deferred.

**meeting_test:** How many tests today mock or patch global model state, and would dependency injection eliminate all of those mocks?

---

## Frame 5 — Cross-Domain Analogy
_How do completely different fields solve structurally analogous problems?_

---

### 5.1 Analogy: Air Traffic Control — Radar Sweep vs. Continuous Track

**title:** Predictive Pre-Scoring via Radar Pattern

**summary:** Air traffic control doesn't wait for a plane to reach a waypoint to check if it's on course — it continuously projects future position from current trajectory. Apply this to card behavior: during quiet periods, pre-compute risk scores for the top-N highest-velocity cards (those approaching thresholds) so that when their next transaction arrives, the score is already cached and the prediction path just confirms a pre-computed result.

**basis:** external: [Speculative execution in CPU design; read-ahead caching in databases; pre-scoring high-risk segments between transactions is used by FICO Falcon fraud system]

**why_it_matters:** Converts the latency problem from "compute on demand" to "confirm pre-computed" for the highest-risk cards, which are also the highest-throughput cards.

**meeting_test:** What percentage of our transaction volume comes from the top-1% highest-velocity cards, and what would pre-scoring those cards cost in background CPU?

---

### 5.2 Analogy: Immune System — Innate vs. Adaptive Response

**title:** Two-Layer Defense: Fast Rules + Slow Model

**summary:** The immune system has two layers: fast innate response (generic rules, fires immediately) and slow adaptive response (specific, trained, takes days). Map this to fraud: layer a fast rule engine (Redis Bloom filter for known-bad card hashes, velocity rules like "3 transactions in 60 seconds") before the LightGBM model. Clear-cut bad actors are caught by the rule engine; ambiguous cases go to the model.

**basis:** external: [Stripe Radar's rule engine + ML model two-layer architecture; Kount's deterministic rule pre-filter; SIEM systems' signature detection before behavioral analytics]

**why_it_matters:** Known-bad cards are caught in microseconds at O(1) cost. The LightGBM model's compute is reserved for genuinely ambiguous cases.

**meeting_test:** How many of our current fraud catches could be captured by simple velocity rules (>3 tx/minute, duplicate amount within 30s), and what fraction require the full model?

---

### 5.3 Analogy: Airplane Black Box — Immutable Write-Ahead Log

**title:** Transaction Write-Ahead Log for Forensic Replay

**summary:** Aircraft flight recorders write every sensor reading to an immutable append-only log, enabling full forensic reconstruction after any incident. Apply this to the prediction pipeline: all raw inputs, feature values, model scores, and decisions should be written to an immutable append-only log (S3, CloudWatch Logs with no delete policy) before the response is returned. Any future model, any retrospective analysis, can replay this log.

**basis:** external: [Write-ahead logging in databases (PostgreSQL WAL); event sourcing pattern; immutable audit trails in PCI DSS compliance]

**why_it_matters:** Enables model retrospectives, regulatory audit compliance, and the ability to re-score historical transactions with a new model without re-running the live pipeline.

**meeting_test:** Do we currently have the ability to re-score yesterday's transactions with today's model, and what would that capability be worth for model validation?

---

### 5.4 Analogy: Supply Chain — Just-In-Time vs. Safety Stock

**title:** Feature Pre-Computation as Safety Stock

**summary:** Supply chains keep safety stock of high-demand items to absorb demand spikes. Apply this: for cards with high transaction velocity (top 1% by transaction count), maintain a "safety stock" of pre-computed features in a Redis Hash, updated on every write. Low-velocity cards can be computed on demand. This is just-in-time vs. make-to-stock, applied to feature computation.

**basis:** external: [Demand-driven supply chain planning; tiered caching in CDN design (hot vs. cold content); selective pre-materialization in Feast based on entity frequency]

**why_it_matters:** The latency problem is worst for high-velocity cards (frequent shoppers, corporate cards). Those are exactly the cards that most benefit from pre-materialized features.

**meeting_test:** Can we identify the top 10K highest-velocity cards and pre-materialize their features without significant Redis memory overhead?

---

### 5.5 Analogy: Chess Engine — Evaluation Function vs. Minimax Depth

**title:** Risk-Stratified Computation Depth

**summary:** Chess engines allocate more computation time to complex positions and use a fast evaluation function for simple ones. Apply this to fraud scoring: transactions with very low or very high raw signals (trivial cases) get a shallow feature set and early exit. Only transactions in the uncertain middle band get the full feature computation + model inference + SHAP.

**basis:** external: [Chess engine selective deepening (quiescence search); multi-armed bandit arm elimination; cascaded classifiers in computer vision]

**why_it_matters:** The computational budget is currently flat across all transactions. Risk-stratified depth makes it proportional to the actual uncertainty of each case.

**meeting_test:** Looking at fraud probability distributions, what fraction of transactions score below 0.02 or above 0.98, where a fast early exit would be safe?

---

### 5.6 Analogy: Epidemiology — Contact Tracing Graph Propagation

**title:** Graph-Propagated Risk Score from Shared Merchant

**summary:** Contact tracing identifies high-risk individuals by proximity to known infected individuals, not just their own symptoms. Apply this to fraud: if card A commits fraud at merchant M, all other cards that transacted with merchant M in the last 24 hours receive an elevated prior risk score (a "contact-traced" risk boost). This catches coordinated fraud rings that share a compromised merchant.

**basis:** external: [Graph-based fraud detection at PayPal (2016 KDD paper); bipartite card-merchant graph propagation; Amazon's fraud graph neural network approach]

**why_it_matters:** Point-in-time card features miss coordinated fraud patterns. Graph propagation catches ring attacks that no individual card's history reveals.

**meeting_test:** Do we have merchant-level data in our transaction history that would allow constructing a card-merchant bipartite graph, even retrospectively?

---

## Frame 6 — Constraint-Flipping
_Invert the obvious constraint to its opposite or extreme._

---

### 6.1 Flip: What If Latency Were Not a Constraint (Unlimited Compute)?

**title:** Full Behavioral Profile Scoring — "No Latency" Design

**summary:** If latency were unlimited, we'd compute 500 features: graph centrality of the card-merchant network, seasonal deviation across 90 days, merchant category risk index, device fingerprint embedding. Flip the current constraint: design this "unlimited" system, then identify which of those features can be pre-computed and cached so they add zero latency. The design exercise reveals which features are currently absent not because they're expensive, but because they were never considered.

**basis:** reasoned: [Constraint relaxation as a design technique (TRIZ inventive principles); "what would we do with infinite compute" is a standard architectural thought experiment that reveals hidden assumptions]

**why_it_matters:** We may be leaving significant predictive power on the table by only considering features that are cheap to compute on demand. Pre-computation unlocks the expensive ones.

**meeting_test:** Which features not in the current model would we add immediately if computation cost were zero, and which of those could be pre-computed?

---

### 6.2 Flip: What If SHAP Ran Before Inference?

**title:** Explanation-First Inference — Pre-compute Explanation Space

**summary:** The constraint is that SHAP runs after inference (as explanation of the result). Flip it: pre-compute the contribution of each feature to the fraud score for a standardized card profile at startup. When a real transaction arrives, compute the delta of each feature from the baseline, scale the pre-computed contributions, and return an approximate but instant explanation with the prediction result — no background task needed.

**basis:** external: [SHAP TreeExplainer `background` parameter for baseline computation; Taylor series approximation of SHAP values from a fixed reference point; linear SHAP approximation for online systems]

**why_it_matters:** Explanation becomes synchronous and sub-millisecond. The accuracy loss from approximation is acceptable for operator use cases (they need direction, not exact values).

**meeting_test:** How much accuracy would we lose in SHAP values by using a pre-computed baseline approximation vs. full TreeSHAP, and is that loss acceptable for our explanation consumers?

---

### 6.3 Flip: What If Card History Were 10,000 Items Instead of 10?

**title:** Deep History with Temporal Decay Weighting

**summary:** The current WINDOW_SIZE=10 cap discards all history beyond 10 transactions. Flip it: keep unlimited history but apply exponential decay weighting (recent transactions count more). Implement this as a running weighted mean maintained in a Redis Hash, updated in O(1) on every write. The "window" concept disappears; behavioral context accumulates indefinitely.

**basis:** external: [Exponentially weighted moving average (EWMA) for streaming statistics; Redis HINCRBYFLOAT for running weighted sum; long-memory models in fraud detection (90-day baseline patterns)]

**why_it_matters:** Fraud patterns that emerge over months (slow-burn account takeovers, incremental credit limit testing) are invisible to a 10-transaction window. Deep history with decay captures them.

**meeting_test:** How much additional fraud would a 90-day behavioral baseline catch that a 10-transaction window misses, and can we test this offline on historical data?

---

### 6.4 Flip: What If the Model Retrained Every Minute?

**title:** Continuous Micro-Training on Confirmed Labels

**summary:** The current model is a static artifact retrained periodically by engineers. Flip it: use the confirmed fraud labels from chargeback data (available with ~24h delay) to continuously fine-tune the model's leaf weights using LightGBM's `update` method on a rolling buffer of confirmed examples. No full retraining — just leaf weight adjustments that adapt to concept drift in near-real-time.

**basis:** external: [LightGBM `model.update()` method for online learning on new data; concept drift adaptation in streaming ML systems; Vowpal Wabbit-style online gradient descent on trees]

**why_it_matters:** Fraud patterns shift weekly. A static model degrades. Micro-training keeps the model current with minimal infrastructure — just a background job that runs `model.update()` on a buffer of new confirmed labels.

**meeting_test:** How quickly do new fraud patterns currently appear and degrade model performance before a manual retrain occurs?

---

### 6.5 Flip: What If We Sent No Response Until We Were Certain?

**title:** Probabilistic Hold Response for High-Uncertainty Transactions

**summary:** The current constraint is: return a decision (allow/block) synchronously within 10ms. Flip it: for transactions in the high-uncertainty band (probability 0.4–0.6), return a "hold" response that triggers a 30-second asynchronous enrichment — fetching additional context (device fingerprint, merchant risk index, graph risk score) — before returning a final decision. High-certainty cases still get sub-10ms responses.

**basis:** external: [Visa's asynchronous risk enrichment workflow for card-not-present transactions; 3D Secure 2.0 "soft decline" pattern for step-up authentication; payment network "referral" response type]

**why_it_matters:** The 10ms constraint forces us to use only features available in 10ms. A 30-second enrichment window for uncertain cases unlocks dramatically better features for exactly the transactions where accuracy matters most.

**meeting_test:** What fraction of our transactions land in the 0.4–0.6 probability band, and what would a 30-second hold mean for the checkout experience at our highest-volume merchants?

---

### 6.6 Flip: What If the API Accepted No Features at All?

**title:** Zero-Input API — Card ID Is Sufficient

**summary:** The current API accepts `card_id`, `amount`, and `hour`. Flip it: what if the API accepted only `card_id` and `amount`, and derived all time features (hour, day_of_week, is_weekend) server-side from the system clock? The caller knows nothing about feature engineering; the API encapsulates all temporal context. This removes an entire class of client bugs (wrong timezone, missing hour, stale timestamp).

**basis:** reasoned: [API contract design: the caller should provide raw business facts (who, how much) not derived signals (what time it is in which timezone); server-side time derivation eliminates client clock skew bugs]

**why_it_matters:** Simpler client contract, fewer integration bugs, and the API can use higher-precision server-side timestamps rather than relying on client-provided hour values.

**meeting_test:** Are there cases where the caller legitimately needs to override the server-side timestamp (e.g., batch reprocessing historical transactions), and how do we handle that exception?
