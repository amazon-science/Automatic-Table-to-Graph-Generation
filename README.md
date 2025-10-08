# AutoG: Towards automatic graph construction from tabular data

## Introduction

## Overview

AutoG is a novel framework that addresses the critical challenge of automatically constructing high-quality graphs from tabular data for graph machine learning (GML) applications. While GML has seen tremendous growth, the crucial step of converting tabular data into meaningful graphs remains largely manual and unstandardized. AutoG leverages Large Language Models (LLMs) to automate this process, producing graphs that rival those created by human experts.

### Key Features

- Automatic graph schema generation without human intervention
- LLM-based solution for high-quality graph construction

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/amazon-science/Automatic-Table-to-Graph-Generation

cd Automatic-Table-to-Graph-Generation/
```

### Step 2: Install Core Dependencies

Recommend using docker to prepare the environment.
``` bash 
docker build -t autog .
```

Then create a mapping for consistent installation 
``` bash
docker run --gpus all -it -v ../Automatic-Table-to-Graph-Generation:/workspace -v ./opt:/opt autog /bin/bash
```

Then you need to configure the LD_LIBRARY_PATH
``` bash
export LD_LIBRARY_PATH="/opt/conda/envs/dbinfer-gpu/lib:$LD_LIBRARY_PATH"
```

### Step 3: Optional Dependencies
These are required for development but not necessary if using cached LLM outputs:

```bash
pip install llama-index-llms-bedrock
pip install llama-index
pip install valentine
```

## Usage

### 1. Dataset Preparation

You need to first configure your kaggle credentials for downloading  the `ieee-cis` dataset


Generate the preprocessing dataset:

```bash
bash scripts/download.sh
```

This creates two dataset versions:
- **Old Version**: A baseline preprocessed version using basic heuristics
- **Expert Version**: Human expert-generated version with optimized column naming

> Note: AutoG uses the 'old' version as input while ignoring the schema information. 

### 2. Running AutoG

To run the AutoG pipeline:

```bash
bash scripts/autog.sh
```

For detailed configuration options, see `scripts/autog.sh`.

### 3. Running Graph Machine Learning

Execute GML tasks on the constructed graphs:

```bash
bash scripts/run.sh
```

### Update: now we stick to a non-api version of the llm, you can directly run the script and then paste the output from your browser-end LLMs for quick testing.

## Using AutoG with Custom Datasets

Follow these steps to apply AutoG to your own data:

1. Generate metadata information:
   ```python
   from models.llm.gconstruct import analyze_dataframes
   metadata = analyze_dataframes(your_dataframe)
   ```

2. Generate initial type predictions:
   ```python
   from prompts import identify
   types = identify(metadata)
   ```

3. Create a DBBRDBDataset wrapper for your data.

4. Generate first-round prompts using AutoG.



## Citation

If you use AutoG in your research, please cite:

```bibtex
@inproceedings{
chen2025autog,
title={AutoG: Towards automatic graph construction from tabular data},
author={Zhikai Chen and Han Xie and Jian Zhang and Xiang song and Jiliang Tang and Huzefa Rangwala and George Karypis},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=hovDbX4Gh6}
}
```




