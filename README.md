# MER2025-DES: Data Augmentation and Mamba-enhanced Temporal Pre-Fusion for Multimodal Affective Explanation

[![Paper](https://img.shields.io/badge/Paper-MER2025-blue)](https://arxiv.org/abs/2504.19423)
[![Task](https://img.shields.io/badge/Task-MER--DES-green)](https://arxiv.org/abs/2504.19423)
[![License](https://img.shields.io/badge/License-Research-orange)](LICENSE)

&gt; **🏆 MER2025 Challenge Solution**  
&gt; This repository contains our official solution for the **MER2025 Multimodal Emotion Recognition with Description Generation (MER-DES)** track.  
&gt; We introduce a novel framework combining **automated reasoning data augmentation** with a **Mamba-based temporal pre-fusion mechanism** to generate human-aligned, explainable emotion descriptions from video, audio, and text.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Qualitative Analysis](#qualitative-analysis)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## 🔍 Overview

Multimodal emotion reasoning requires fine-grained alignment across visual, auditory, and textual cues. Our framework addresses two core challenges in MER-DES:

1. **Data Scarcity**: The only human-annotated reasoning dataset (OV-MERD) contains merely **332 samples**, insufficient for training Multimodal Large Language Models (MLLMs).
2. **Temporal Dynamics Loss**: Standard tokenization and self-attention mechanisms in LLMs often discard fine-grained emotional evolutions in temporally evolving audio-visual streams.

### Our Solution

| Component | Description |
|-----------|-------------|
| **Data Engine** | Fine-tuned AffectGPT on OV-MERD → generate reasoning on SIMSv2 → **Qwen3 consistency filtering** → **human revision** → **4,403 high-quality samples** |
| **MTF Module** | **Mamba-enhanced Temporal pre-Fusion** capturing long-range temporal dependencies and cross-modal gating before LLM decoding |
| **Backbone** | Qwen2.5-7B-Instruct with frozen SigLIP-SO400M (visual) and HuBERT-L (audio) encoders |

---

## ✨ Key Features

- 🎯 **Human-Aligned Reasoning**: Two-stage quality control (automatic + manual) ensures generated training data reflects natural human inference processes.
- 🧠 **Mamba Temporal Fusion**: Parallel SSM streams with cross-modal gating explicitly model emotional dynamics across audio and visual modalities.
- 🔗 **Cross-Modal Alignment**: Attention-weighted pooling + gated interaction prevents fine-grained signal loss during tokenization.
- 📊 **State-of-the-Art Performance**: Significantly outperforms AffectGPT on emotion description coherence and interpretability.

---

## 🏗️ Architecture
