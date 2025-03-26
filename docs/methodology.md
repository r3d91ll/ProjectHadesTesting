# Experimental Methodology

This document outlines the detailed methodology for conducting the XnX notation validation experiments across all three phases.

## Common Experimental Framework

All experiments will follow these general guidelines to ensure consistency and reproducibility:

### Datasets

We will use the exact datasets specified in the original PathRAG and GraphRAG papers:
- KILT datasets (Natural Questions, HotpotQA, etc.)
- Any additional datasets used in the original papers

### Metrics

The following metrics will be collected consistently across all experiments:

1. **Retrieval Quality Metrics**
   - Precision@k (k=1,3,5)
   - Recall@k (k=1,3,5)
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (nDCG)

2. **Answer Quality Metrics**
   - ROUGE-L score
   - BLEU score
   - BERTScore
   - Human evaluation scores (when applicable)

3. **Reasoning Transparency Metrics**
   - Path explainability score (qualitative)
   - Confidence correlation (correlation between model confidence and answer correctness)

4. **Efficiency Metrics**
   - Latency (ms)
   - Memory usage
   - Throughput (queries per second)

### Run Configuration

- Each experiment will be run 5 times to account for variance
- Statistical significance will be calculated using paired t-tests
- 95% confidence intervals will be reported for all metrics

## Phase-Specific Methodologies

### Phase 1: Original Implementation Verification

1. **Setup Steps**
   - Clone original PathRAG and GraphRAG repositories
   - Set up dependencies exactly as specified in the papers
   - Configure embedding models to match paper specifications
   - Initialize database systems (ArangoDB for PathRAG, Neo4j for GraphRAG)

2. **Verification Process**
   - Run the original implementations on the specified datasets
   - Compare results with those reported in the papers
   - Document any discrepancies and account for them
   - Establish baseline metrics for subsequent phases

3. **Success Criteria**
   - Results within 5% of reported metrics in the papers
   - Stable performance across multiple runs
   - Complete documentation of the experimental environment

### Phase 2: Qwen2.5 Coder Integration

1. **Model Integration Steps**
   - Set up Ollama with Qwen2.5 Coder model
   - Create adapter layer to replace original models with Qwen2.5
   - Ensure prompt formats are consistent with original experiments
   - Validate adapter functionality with simple test cases

2. **Testing Process**
   - Run identical experiments from Phase 1 with only the model changed
   - Maintain all other parameters and configurations
   - Document differences in model behavior and output quality
   - Compare performance metrics with Phase 1 baselines

3. **Success Criteria**
   - Establish stable performance with Qwen2.5 Coder
   - Document any performance differences from Phase 1
   - Identify model-specific strengths and weaknesses

### Phase 3: XnX Notation Integration

1. **XnX Integration Steps**
   - Implement XnX notation formatting for graph paths
   - Integrate notation into context construction for both PathRAG and GraphRAG
   - Ensure consistent application of notation weights and structural patterns
   - Create modified prompts that leverage XnX notation

2. **Testing Process**
   - Run experiments with both original models and Qwen2.5 Coder
   - Test various XnX formatting approaches to identify optimal configurations
   - Collect both quantitative metrics and qualitative assessment of output quality
   - Compare results against both Phase 1 and Phase 2 baselines

3. **Success Criteria**
   - Quantifiable improvement in at least one key metric
   - Consistent performance gains across multiple datasets
   - Clear demonstration of XnX notation's impact on output quality

## Analysis and Documentation

For each phase, we will produce:

1. **Detailed Results Tables**
   - All metrics with confidence intervals
   - Percentage differences from baselines
   - Statistical significance indicators

2. **Visualizations**
   - Performance comparison charts
   - Error analysis visualizations
   - Path quality visualization for XnX notation

3. **Case Studies**
   - Examples of queries where XnX notation improves results
   - Analysis of failure cases
   - Comparative analysis of path quality

4. **Discussion and Interpretation**
   - Analysis of why XnX notation impacts performance
   - Identification of specific use cases where XnX excels
   - Recommendations for future research and development

## Reproducibility Guidelines

All experiments will be documented with:
- Full environment specifications
- Exact configuration parameters
- Code versions and dependencies
- Random seeds for reproducibility
- Complete data processing pipelines

This ensures that all results can be independently verified and reproduced.
