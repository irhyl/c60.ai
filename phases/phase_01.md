Always wrap code in triple backticks with language specified.

Phase 1: Project Setup & Design (Steps 1–50)

1. Define C60.ai's core vision and identity.
Clarify that C60.ai will be an AutoML system that builds self-evolving machine learning pipelines using molecular graph concepts and meta-learning. Its pipelines are directed acyclic graphs (DAGs) of transformations and models. Define guiding principles in /docs/manifesto.md. C60's core logic will reside in /engine, metadata and graphs in /memory, visual demos in /notebooks. This clarity informs architectural choices and avoids overengineering.

2. Select and document the complete tech stack.
Use Python as the base language. Include core libraries: scikit-learn for ML models, networkx for DAGs, pytorch and torch_geometric for GNN-based evaluations, optuna for hyperparameter optimization, and mlflow for experiment tracking. Store all dependency versions in environment.yml. Include Jupyter notebooks in /notebooks to document and demonstrate usage. Log stack decisions in docs/tech_stack.md for contributors.

3. Initialize the code repository and file system.
Structure the repo as follows:

/engine: Core logic (search, graph builder, mutation, scoring)

/memory: Meta-learning storage and pipeline results

/notebooks: Interactive notebooks for experiments and tutorials

/interface: CLI and Streamlit UI

/datasets: Sample datasets and loaders

/tests: Unit and integration tests
Initialize Git, create .gitignore, and commit this structure to GitHub.

4. Set up reproducible development environment.
Use Conda to manage dependencies. Create an environment c60 with Python 3.11. Save and export dependencies using conda env export > environment.yml. Include jupyterlab, black, and pytest for development. Add a Makefile to automate commands like make test, make format, and make run. Document this in docs/dev_setup.md.

5. Implement dataset loader and validator.
Create /engine/data_loader.py with load_csv(filepath) function. It returns a DataFrame and metadata (feature types, null counts, cardinality, etc.). Infer whether the task is classification or regression using target type. Also write validate_dataset(df) to ensure data integrity. Store test files in /datasets/test_iris.csv. Create /notebooks/1_dataset_loading.ipynb to demo this.

6. Define pipeline graph schema and storage format.
In /engine/graph_schema.py, define Node and Edge classes for pipeline components. Each Node includes name, type (e.g., Preprocessor, Model), and parameters. Each Edge includes source, destination, and validity. Store graphs as JSON under /memory/graphs/, using a schema version (v1.0). Visualize sample graphs using networkx.draw() in /notebooks/2_graph_structure.ipynb.

7. Build a graph validator module.
In /engine/graph_validator.py, implement validate_graph(graph) that checks for DAG structure (acyclic), correct input-output flow, and compatibility between consecutive nodes (e.g., scaler followed by model). Include error messages. Add test cases to /tests/test_validator.py. In /notebooks/3_graph_validation.ipynb, demonstrate valid and invalid pipelines.

8. Implement base ML components (nodes).
Create /engine/nodes.py defining base node templates:

PreprocessorNode: Imputer, Scaler, Encoder

FeatureNode: PCA, SelectKBest

ModelNode: RandomForest, SVM, XGBoost, LightGBM
Include hyperparameter schemas (param ranges, types). These are the base units C60 evolves. In /notebooks/4_node_templates.ipynb, list and visualize them.

9. Create graph mutation functions.
In /engine/mutator.py, implement mutation functions:

add_node(graph): inserts node and reconnects DAG

remove_node(graph): deletes a node safely

mutate_params(graph): tweaks hyperparameters
Design mutations with constraints to avoid graph corruption. Mutation logic must keep graphs valid. Demonstrate these mutations in /notebooks/5_graph_mutations.ipynb.

10. Develop the graph-to-pipeline compiler.
In /engine/compiler.py, write compile_graph(graph) to convert the DAG into a sklearn.Pipeline object. Ensure node order matches dependencies. Log invalid pipelines during compilation. Add fallback if conversion fails. In /notebooks/6_compile_pipeline.ipynb, visualize pipeline steps.

11. Implement feature summarization engine.
In /engine/feature_summarizer.py, create summarize_features(df) that calculates per-column stats: min, max, mean, variance, missing %, skew, kurtosis, and entropy for categorical features. This forms the base for dataset fingerprinting. Use scipy.stats.entropy for information content. Save outputs as a summary dict in JSON. Create /notebooks/7_feature_summary.ipynb to show summaries for test datasets. This module will feed into both meta-learning and graph-guided mutation later.

12. Design fixed-length dataset fingerprint vector.
Create a vector embedding of dataset properties in /engine/fingerprint.py. Define a consistent 64-dimensional structure combining statistics from feature summarizer. Normalize features using min-max scaling. Apply PCA if needed to reduce dimensionality. Store fingerprint in /memory/fingerprints/{dataset_name}.json. Include fingerprint-to-dataset mapping for warm starts. Add /notebooks/8_fingerprint_generation.ipynb to explain the structure and visualize embeddings using t-SNE.

13. Build persistent memory module.
In /memory/memory_store.py, implement MemoryStore class with CRUD methods: save_pipeline_result(fingerprint, graph, score), get_similar(fingerprint), list_all(). Use sqlite3 or flat JSON as backend. Ensure memory can be queried by cosine similarity of fingerprints. Integrate schema validation to prevent malformed graph insertions. In /notebooks/9_memory_module.ipynb, demo how memory retrieves past successful pipelines.

14. Define pipeline performance scoring function.
Create /engine/scorer.py. Implement score_pipeline(pipeline, X, y, task) using cross-validation (5-fold for classification, TimeSeriesSplit for time series). Use accuracy, f1, and roc_auc as default metrics for classification, and r2, rmse for regression. Handle exceptions (e.g., fitting errors). Return score and standard deviation. In /notebooks/10_pipeline_scoring.ipynb, test this on baseline pipelines.

15. Implement base search loop.
In /engine/search_loop.py, build C60SearchLoop class. It initializes with dataset fingerprint, queries memory, and begins mutation cycles from best prior graph or a default seed. For each generation, mutate, compile, score, and retain best N. Track stats per generation: score delta, mutation type, and run time. Log everything in run_log.json. Add /notebooks/11_search_loop_demo.ipynb to illustrate full search iterations.

16. Enable early stopping in search.
Inside C60SearchLoop, implement early stopping logic. Monitor moving average of scores over last 5 generations. If improvement < threshold (e.g., 0.001), halt search. Use patience parameter and max trials. Track convergence time. Store early stopping flags in logs. Visualize generation-vs-score trend in /notebooks/12_early_stop_analysis.ipynb. Helps reduce wasted computation during optimization.

17. Develop graph mutation history tracker.
In /engine/mutation_history.py, define MutationTracker class that stores lineage of every mutated graph. Each entry links parent → child with mutation type, parameters changed, and performance delta. Output as mutation tree JSON. Visualize using networkx tree drawing. Add /notebooks/13_mutation_tree.ipynb to explore evolution paths of top-performing pipelines.

18. Add mutation scoring heuristic.
In /engine/mutator.py, enhance mutate(graph) with optional heuristic=True flag. Use recent mutation success history (e.g., param tweaks led to 60% improvement) to guide future mutation type probabilities. Update these heuristics in-memory after each trial. Log frequency and outcomes per mutation type. Explain adaptive mutation tuning in /notebooks/14_mutation_heuristics.ipynb.

19. Encode pipelines using molecular graph structure.
Represent pipeline DAGs as molecule-like graphs in /engine/graph_encoder.py. Each node = functional unit (e.g., scaler), each edge = data transformation. Assign feature vectors to nodes: operation type, complexity (e.g., O(n log n)), tunability, past success rate. Use this for future GNN input. Add /notebooks/15_graph_encoding.ipynb to describe encoding logic and visualize graph matrices.

20. Train graph neural network (GNN) predictor.
Use torch_geometric to define a GNN in /engine/gnn_predictor.py that predicts pipeline performance (regression: expected score). Input: pipeline graph, output: score estimate. Train using previously evaluated pipelines stored in memory. Use GraphConv layers and MSE loss. Evaluate prediction quality via R2. Demonstrate full training loop in /notebooks/16_gnn_pipeline_predictor.ipynb.

21. Integrate GNN into mutation selection.
Modify mutate() function to score each child graph with GNN before actual evaluation. Select top-k predicted graphs for actual scoring. This enables pruning of poor configurations. Add a GNN cache to avoid re-encoding. Explain this intelligent pruning in /notebooks/17_gnn_guided_search.ipynb. This is key to C60’s efficiency and novelty.

22. Build fingerprint-similarity based warm start.
In C60SearchLoop, on new dataset load, compare fingerprint against stored dataset fingerprints using cosine similarity. If match > 0.9, extract best past pipelines as initial candidates. This mimics meta-learning. Store this similarity search logic in /engine/meta_learner.py. Visualize fingerprint similarity matrix in /notebooks/18_meta_learning_start.ipynb.

23. Implement pipeline replay and audit.
In /interface/replayer.py, allow user to load any pipeline JSON from memory and re-run it on new data. Audit node-by-node transformations, timings, and errors. Enable flag to visualize intermediate outputs (e.g., post-scaling features). Demo this pipeline replayer in /notebooks/19_pipeline_replay.ipynb. Critical for reproducibility and debugging.

24. Build CLI tool to run search jobs.
Create /interface/cli.py. Enable arguments: --file, --task, --max_trials, --guided, --export. Parse arguments with argparse. Start full C60 loop and output leaderboard. Store CLI results in outputs/. Add example in /notebooks/20_cli_demo.ipynb showing programmatic vs command-line usage.

25. Add output visualizations and reports.
In /interface/visualizer.py, build charts:

Score per generation

Mutation impact heatmap

Pipeline structure plot
Use matplotlib and networkx. Export results as report.html post-run. Demo all outputs in /notebooks/21_report_outputs.ipynb. This improves UX for both academic and practical audiences.

26. Store mutation-provenance lineage tree.
Link all pipelines into a tree structure in /memory/lineages/. Each node includes metadata: mutation, parent ID, score. Export as .dot or .json and render using Graphviz. This lineage tree is a central artifact of C60’s evolutionary learning. Walk through lineage generation in /notebooks/22_lineage_tree.ipynb.

27. Add pipeline serialization and export module.
In /engine/exporter.py, serialize any sklearn-compatible pipeline into .pkl, .json, and .py formats. This allows future loading without retraining. Add CLI flag --export-best to save top pipeline. Demo export/import round trip in /notebooks/23_pipeline_export.ipynb.

28. Implement mutation-type performance analyzer.
Build /engine/mutation_stats.py to track each mutation type’s average delta score and success rate. Maintain rolling windows to update weights. Use this to optionally bias future mutation sampling. Visualize mutation efficacy trends in /notebooks/24_mutation_stats.ipynb.

29. Add configuration manager.
In /engine/config.py, define a singleton ConfigManager to load runtime parameters from config.yaml. Include defaults: max_trials, scoring metric, logging paths. Load configs at start of search. Create /notebooks/25_config_tuning.ipynb to show how configuration impacts search results.

30. Develop test suite for core modules.
Write tests for all classes: GraphValidator, Compiler, Mutator, Scorer, SearchLoop. Place in /tests/. Use pytest. Include synthetic data fixtures in /tests/fixtures/. Validate pipeline legality, mutation coverage, and scoring consistency. Document all test cases in /notebooks/26_test_suite_demo.ipynb.

11. Implement feature summarization engine.
In /engine/feature_summarizer.py, create summarize_features(df) that calculates per-column stats: min, max, mean, variance, missing %, skew, kurtosis, and entropy for categorical features. This forms the base for dataset fingerprinting. Use scipy.stats.entropy for information content. Save outputs as a summary dict in JSON. Create /notebooks/7_feature_summary.ipynb to show summaries for test datasets. This module will feed into both meta-learning and graph-guided mutation later.

12. Design fixed-length dataset fingerprint vector.
Create a vector embedding of dataset properties in /engine/fingerprint.py. Define a consistent 64-dimensional structure combining statistics from feature summarizer. Normalize features using min-max scaling. Apply PCA if needed to reduce dimensionality. Store fingerprint in /memory/fingerprints/{dataset_name}.json. Include fingerprint-to-dataset mapping for warm starts. Add /notebooks/8_fingerprint_generation.ipynb to explain the structure and visualize embeddings using t-SNE.

13. Build persistent memory module.
In /memory/memory_store.py, implement MemoryStore class with CRUD methods: save_pipeline_result(fingerprint, graph, score), get_similar(fingerprint), list_all(). Use sqlite3 or flat JSON as backend. Ensure memory can be queried by cosine similarity of fingerprints. Integrate schema validation to prevent malformed graph insertions. In /notebooks/9_memory_module.ipynb, demo how memory retrieves past successful pipelines.

14. Define pipeline performance scoring function.
Create /engine/scorer.py. Implement score_pipeline(pipeline, X, y, task) using cross-validation (5-fold for classification, TimeSeriesSplit for time series). Use accuracy, f1, and roc_auc as default metrics for classification, and r2, rmse for regression. Handle exceptions (e.g., fitting errors). Return score and standard deviation. In /notebooks/10_pipeline_scoring.ipynb, test this on baseline pipelines.

15. Implement base search loop.
In /engine/search_loop.py, build C60SearchLoop class. It initializes with dataset fingerprint, queries memory, and begins mutation cycles from best prior graph or a default seed. For each generation, mutate, compile, score, and retain best N. Track stats per generation: score delta, mutation type, and run time. Log everything in run_log.json. Add /notebooks/11_search_loop_demo.ipynb to illustrate full search iterations.

16. Enable early stopping in search.
Inside C60SearchLoop, implement early stopping logic. Monitor moving average of scores over last 5 generations. If improvement < threshold (e.g., 0.001), halt search. Use patience parameter and max trials. Track convergence time. Store early stopping flags in logs. Visualize generation-vs-score trend in /notebooks/12_early_stop_analysis.ipynb. Helps reduce wasted computation during optimization.

17. Develop graph mutation history tracker.
In /engine/mutation_history.py, define MutationTracker class that stores lineage of every mutated graph. Each entry links parent → child with mutation type, parameters changed, and performance delta. Output as mutation tree JSON. Visualize using networkx tree drawing. Add /notebooks/13_mutation_tree.ipynb to explore evolution paths of top-performing pipelines.

18. Add mutation scoring heuristic.
In /engine/mutator.py, enhance mutate(graph) with optional heuristic=True flag. Use recent mutation success history (e.g., param tweaks led to 60% improvement) to guide future mutation type probabilities. Update these heuristics in-memory after each trial. Log frequency and outcomes per mutation type. Explain adaptive mutation tuning in /notebooks/14_mutation_heuristics.ipynb.

19. Encode pipelines using molecular graph structure.
Represent pipeline DAGs as molecule-like graphs in /engine/graph_encoder.py. Each node = functional unit (e.g., scaler), each edge = data transformation. Assign feature vectors to nodes: operation type, complexity (e.g., O(n log n)), tunability, past success rate. Use this for future GNN input. Add /notebooks/15_graph_encoding.ipynb to describe encoding logic and visualize graph matrices.

20. Train graph neural network (GNN) predictor.
Use torch_geometric to define a GNN in /engine/gnn_predictor.py that predicts pipeline performance (regression: expected score). Input: pipeline graph, output: score estimate. Train using previously evaluated pipelines stored in memory. Use GraphConv layers and MSE loss. Evaluate prediction quality via R2. Demonstrate full training loop in /notebooks/16_gnn_pipeline_predictor.ipynb.

21. Integrate GNN into mutation selection.
Modify mutate() function to score each child graph with GNN before actual evaluation. Select top-k predicted graphs for actual scoring. This enables pruning of poor configurations. Add a GNN cache to avoid re-encoding. Explain this intelligent pruning in /notebooks/17_gnn_guided_search.ipynb. This is key to C60’s efficiency and novelty.

22. Build fingerprint-similarity based warm start.
In C60SearchLoop, on new dataset load, compare fingerprint against stored dataset fingerprints using cosine similarity. If match > 0.9, extract best past pipelines as initial candidates. This mimics meta-learning. Store this similarity search logic in /engine/meta_learner.py. Visualize fingerprint similarity matrix in /notebooks/18_meta_learning_start.ipynb.

23. Implement pipeline replay and audit.
In /interface/replayer.py, allow user to load any pipeline JSON from memory and re-run it on new data. Audit node-by-node transformations, timings, and errors. Enable flag to visualize intermediate outputs (e.g., post-scaling features). Demo this pipeline replayer in /notebooks/19_pipeline_replay.ipynb. Critical for reproducibility and debugging.

24. Build CLI tool to run search jobs.
Create /interface/cli.py. Enable arguments: --file, --task, --max_trials, --guided, --export. Parse arguments with argparse. Start full C60 loop and output leaderboard. Store CLI results in outputs/. Add example in /notebooks/20_cli_demo.ipynb showing programmatic vs command-line usage.

25. Add output visualizations and reports.
In /interface/visualizer.py, build charts:

Score per generation

Mutation impact heatmap

Pipeline structure plot
Use matplotlib and networkx. Export results as report.html post-run. Demo all outputs in /notebooks/21_report_outputs.ipynb. This improves UX for both academic and practical audiences.

26. Store mutation-provenance lineage tree.
Link all pipelines into a tree structure in /memory/lineages/. Each node includes metadata: mutation, parent ID, score. Export as .dot or .json and render using Graphviz. This lineage tree is a central artifact of C60’s evolutionary learning. Walk through lineage generation in /notebooks/22_lineage_tree.ipynb.

27. Add pipeline serialization and export module.
In /engine/exporter.py, serialize any sklearn-compatible pipeline into .pkl, .json, and .py formats. This allows future loading without retraining. Add CLI flag --export-best to save top pipeline. Demo export/import round trip in /notebooks/23_pipeline_export.ipynb.

28. Implement mutation-type performance analyzer.
Build /engine/mutation_stats.py to track each mutation type’s average delta score and success rate. Maintain rolling windows to update weights. Use this to optionally bias future mutation sampling. Visualize mutation efficacy trends in /notebooks/24_mutation_stats.ipynb.

29. Add configuration manager.
In /engine/config.py, define a singleton ConfigManager to load runtime parameters from config.yaml. Include defaults: max_trials, scoring metric, logging paths. Load configs at start of search. Create /notebooks/25_config_tuning.ipynb to show how configuration impacts search results.

30. Develop test suite for core modules.
Write tests for all classes: GraphValidator, Compiler, Mutator, Scorer, SearchLoop. Place in /tests/. Use pytest. Include synthetic data fixtures in /tests/fixtures/. Validate pipeline legality, mutation coverage, and scoring consistency. Document all test cases in /notebooks/26_test_suite_demo.ipynb.

31. Build GNN score cache system.
In /engine/gnn_cache.py, implement a GNNScoreCache to store GNN-predicted scores for unique graph hashes. Use SHA256 or a custom graph hash function. This avoids redundant GNN inference on previously seen graphs. Before evaluating a graph, check the cache. Store the cache as gnn_score_cache.json. Add lookup performance tests in /notebooks/27_gnn_cache_demo.ipynb.

32. Design graph hash encoder.
In /engine/graph_encoder.py, add hash_graph(graph) which flattens node types and edges into a canonical string then hashes it (SHA256). Ensures identical graphs always yield the same ID. This is used for cache lookups and graph comparisons. Demonstrate consistent hash generation in /notebooks/28_graph_hashing.ipynb.

33. Add GNN pretraining phase.
Before user pipelines are run, pretrain GNN model on synthetic graphs. Generate N=1000 graphs with random mutation logic and synthetic scores (score = node count × randomness + noise). Store in pretrain_data/. Use it to bootstrap GNN with general topology-score mapping. Describe this method in /notebooks/29_gnn_pretraining.ipynb.

34. Integrate Optuna hyperparameter tuning.
In /engine/optimizer.py, wrap pipeline evaluation in an Optuna study for optimizing node-level hyperparameters. Allow graph to mutate structure while Optuna tunes params. Log trials using optuna-dashboard. Demo this integration in /notebooks/30_optuna_tuning.ipynb. Adds sophisticated local search.

35. Implement retry logic on pipeline failure.
In search_loop.py, wrap compile+fit inside a try-except. On failure, retry N times with small mutation tweak. Log reason of failure. Save a blacklist of consistently failing subgraphs in memory/failure_patterns.json. Explain fault tolerance design in /notebooks/31_pipeline_failures.ipynb.

36. Generate pipeline blueprints.
After search ends, extract reusable patterns from best pipelines. Save as templates (blueprints/). In /engine/blueprint_generator.py, detect frequent subgraph motifs and annotate them with tags: "scaling→pca→rf". Provide blueprint previews in /notebooks/32_blueprints.ipynb. Used for warm-starting.

37. Score pipeline cost-efficiency.
Add runtime and memory usage tracking to each pipeline trial. In scorer.py, log fit_time, predict_time, RAM_peak. Score = weighted sum of performance and efficiency. Normalize scores across runs. Add a toggle --efficiency-aware in CLI. Explain metric blending in /notebooks/33_efficiency_scores.ipynb.

38. Add ensemble pipeline support.
In graph_schema.py, allow a special EnsembleNode with subgraphs as children. Evaluate each and combine via voting or averaging. In /engine/ensemble_builder.py, implement logic to generate ensembles from top-N pipelines. Explore ensemble impact in /notebooks/34_ensemble_pipelines.ipynb.

39. Visualize pipeline DAG interactively.
In /interface/visualizer.py, use pyvis to create interactive HTML DAGs. Each node shows type, score, parameters. Export to outputs/graph_vis/. Demo full graph exploration experience in /notebooks/35_interactive_graphs.ipynb. Useful for papers and presentations.

40. Track feature importances per pipeline.
In /engine/feature_tracker.py, extract feature_importances_ from compatible models (e.g., RF, XGB). Log which features dominate across top pipelines. Aggregate across trials and plot using seaborn. Include feature impact chart in /notebooks/36_feature_analysis.ipynb.

41. Add pipeline reproducibility checker.
Implement /engine/reproducibility.py to rerun any saved pipeline on the original dataset using a fixed seed. Compare metrics, runtime, and log drift if any. Store reproducibility reports in outputs/reproducibility_logs/. Create reproducibility notebook: /notebooks/37_reproducibility_check.ipynb.

42. Generate automated leaderboard.
After every run, generate a table of top-N pipelines: score, size, runtime, graph hash. Save as outputs/leaderboard.csv and HTML. Include search metadata (dataset, fingerprint, run time). Automate leaderboard generation in /engine/postprocessing.py. Visualize leaderboard evolution in /notebooks/38_leaderboard_history.ipynb.

43. Benchmark C60 against Auto-sklearn.
Load benchmark datasets (from openml or sklearn.datasets). Run C60 and autosklearn under the same constraints (e.g., time, CPU). Log comparison metrics. Store all runs in /benchmarks/. Document methodology in /notebooks/39_benchmark_vs_autosklearn.ipynb. Necessary for paper publication.

44. Handle categorical encoding adaptively.
In nodes.py, support multiple encoders: OneHot, Ordinal, TargetEncoding. Use feature summarizer to choose the best one based on cardinality and distribution. Add rule-based selection logic. Visualize encoder impact on performance in /notebooks/40_categorical_encoders.ipynb.

45. Simulate time series pipelines.
Add support for time-aware preprocessing: lag features, rolling stats, differencing. Implement TimeSeriesNode types. Create test dataset (e.g., airline passengers). Add a time series mode in CLI. Explore this mode in /notebooks/41_time_series_mode.ipynb. Expands C60’s scope.

46. Build a pipeline explanation engine.
In /engine/explain.py, walk through pipeline graph and explain each transformation, purpose, and hyperparameter. Use sklearn’s model inspection tools to explain final predictions. Export as pipeline_explanation.txt. Demo explanation engine in /notebooks/42_pipeline_explainer.ipynb.

47. Track memory growth and metadata.
Log size of memory DB, count of unique graphs, mutation frequency over time. In /memory/monitor.py, emit weekly usage stats. Store in memory/logs/. Chart memory evolution in /notebooks/43_memory_growth.ipynb. Useful for long-term scalability tracking.

48. Add a safety validator module.
Prevent pipelines from using harmful or nonsensical configurations (e.g., PCA before imputation). In /engine/safety_validator.py, implement rule-based and ML-learned constraints. Raise exceptions or auto-fix. Explain safety strategies in /notebooks/44_pipeline_safety.ipynb.

49. Integrate git-based experiment tracking.
Auto-create a git tag for every experiment run. Save pipeline, config, fingerprint, and scores under that tag. Allow restoring past runs using git checkout. Add experiment restore script in /interface/git_restore.py. Demo tagging and recovery in /notebooks/45_git_tracking.ipynb.

50. Finalize Phase 1 deliverables.
Push all modules to GitHub. Run CI tests. Ensure all notebooks from step 1 to 50 exist, are clean, and reproducible. Tag this version as v0.1-alpha. Write a SUMMARY.md linking all modules and notebooks. Record screencast walk-through of C60 up to this stage. You are now ready to begin Phase 2: LLM reasoning, causal constraints, and scaling.