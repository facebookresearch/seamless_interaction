<div align="center">

# Seamless Interaction Dataset

<img src="./assets/banner.png" alt="Seamless Interaction Dataset Banner" width="800px">

**A large-scale multimodal dataset of 4,000+ hours of human interactions for AI research**

[📄 Paper](#citation) | [🤗 HuggingFace](https://huggingface.co/datasets/seamless/interaction) | [🎮 Demo](https://seamless-interaction.github.io/demo) | [📊 Benchmarks](#benchmarks) | [📚 Documentation](https://seamless-interaction.github.io/docs)

</div>

## Overview

Human communication involves a complex interplay of verbal and nonverbal signals, essential for
conveying meaning and achieving interpersonal goals. To develop socially intelligent AI technologies, it is crucial to build models that can both comprehend and generate dyadic behavioral dynamics. 

The **Seamless Interaction Dataset** is a large-scale collection of over 4,000 hours of face-to-face interaction footage from more than 4,000 participants in diverse contexts. This dataset enables the development of AI technologies that understand dyadic embodied dynamics, unlocking breakthroughs in:

- 🤖 Virtual agents and embodied AI
- 🎭 Natural human-computer interaction
- 📡 Advanced telepresence experiences
- 📊 Multimodal content analysis tools
- 🎬 Animation and synthetic content generation

## 🚀 Quick Start

### Installation

```bash
# Install the package
pip install seamless-interaction

# Download a specific dataset version
seamless-interaction download --subset 10GB  # Options: 10GB, 100GB, 1TB, full
```

### Basic Usage

```python
# Import and load the dataset 
from seamless_interaction import load_dataset

# Load the train split with specific modalities
dataset = load_dataset(
    split="train", 
    modalities=["audio", "video", "transcript", "movement"],
    subset="10GB"  # Use the 10GB subset
)

# Iterate through examples
for example in dataset:
    print(f"Interaction ID: {example['id']}")
    print(f"Duration: {example['duration']}s")
    print(f"Transcript: {example['transcript'][:100]}...")
    
    # Access rich movement features
    if 'movement' in example:
        print(f"Emotion scores: {example['movement']['emotion_scores'].shape}")
```

### Using HuggingFace Datasets

```python
# Alternatively, load directly from HuggingFace
from datasets import load_dataset

# Load the 10GB sample from HuggingFace
dataset = load_dataset("seamless/interaction", "10GB")

# Stream the dataset to avoid downloading everything at once
dataset = load_dataset("seamless/interaction", "10GB", streaming=True)
```

## 🔍 Description

The Seamless Interaction repository is split into several main components:

### 📊 Dataset

The repository provides comprehensive tools for downloading, processing, and utilizing the Seamless Interaction dataset for research and development. The dataset includes:

- **Raw and processed multimodal data**: Video, audio, transcripts, and annotations
- **Precomputed features**: Motion capture, facial keypoints, voice activity detection
- **Metadata**: Participant demographics, interaction contexts, and relationships
- **Benchmark tasks**: Standardized evaluation protocols for various AI tasks

### 📂 Repository Structure

```
seamless_interaction/
├── data/                 # Main dataset directory
│   ├── README.md         # Dataset-specific documentation
│   ├── improvised/       # Interactions with guided prompts
│   └── naturalistic/     # Spontaneous conversations
├── sample/               # Sample datasets of different sizes
│   ├── 10GB/             # Mini sample (100 interactions)
│   ├── 100GB/            # Small sample (1,000 interactions)
│   └── 1TB/              # Medium sample (10,000 interactions)
├── scripts/              # Utility scripts for dataset processing
│   ├── constants.py      # Dataset constants and configuration
│   ├── errors.py         # Error handling utilities
│   └── utils.py          # General utility functions
├── LICENSE               # CC-BY-NC 4.0 license
└── pyproject.toml        # Python package configuration
```

### 🛠️ Utility Scripts

The `scripts` directory contains utilities to help process and validate the dataset:

- **constants.py**: Defines dataset paths, modality names, and configuration parameters
- **errors.py**: Custom exceptions and error handling for data loading issues
- **utils.py**: Helper functions for data processing, conversion, and visualization

### 🧠 Imitator Model

The repository also includes the Imitator model architecture, designed to [TBD]


## 📦 Deep Dive into the Dataset

### Dataset Structure

The Seamless Interaction Dataset is organized into two main categories:
- **Improvised**: Interactions based on predefined scenarios with guided prompts
- **Naturalistic**: Spontaneous conversations without predetermined scripts

```
seamless_interaction
├── LICENSE                   # CC-BY-NC 4.0 license file
├── interactions.csv          # Maps interaction_id to prompt_id with all prompt metadata
├── participants.csv          # Contains all metadata on participants
├── relationships.csv         # Contains metadata on participant relationships in each session
├── improvised                # Interactions with guided prompts
│   ├── dev
│   │   ├── 1P-IS/            # xxx
│   │   │   └── V<vendor>_S<session>_I<interaction>_P<participant>.json
│   │   ├── 1P-R/             # xxx
│   │   │   └── V<vendor>_S<session>_I<interaction>_P<participant>.json
│   │   ├── 3P-IS/            # xxx
│   │   │   └── V<vendor>_S<session>_I<interaction>_P<participant>.json
│   │   ├── 3P-R/             # xxx
│   │   │   └── V<vendor>_S<session>_I<interaction>_P<participant>.json
│   │   ├── 3P-V/             # xxx
│   │   │   └── V<vendor>_S<session>_I<interaction>_P<participant>.json
│   │   ├── audio/            # Speaker-bleed denoised audio
│   │   │   └── V<vendor>_S<session>_I<interaction>_P<participant>.wav
│   │   ├── boxes_and_keypoints/
│   │   │   ├── box/          # Bounding boxes for each participant
│   │   │   ├── is_valid_box/ # Whether bounding boxes are valid
│   │   │   └── keypoints/    # Detected facial/body keypoints
│   │   ├── movement/         # Quantified movement features 
│   │   │   ├── emotion_arousal/           # Arousal measures
│   │   │   ├── emotion_valence/           # Valence measures
│   │   │   ├── emotion_scores/            # Emotion detection scores
│   │   │   ├── expression/                # Facial expression parameters
│   │   │   ├── FAUToken/                  # Facial Action Unit tokens
│   │   │   ├── FAUValue/                  # Facial Action Unit values
│   │   │   ├── gaze_encodings/            # Eye gaze direction encodings
│   │   │   ├── head_encodings/            # Head position/rotation encodings
│   │   │   ├── frame_latent/              # Per-frame latent representations
│   │   │   └── is_valid/                  # Validity flags for extracted features
│   │   ├── smplh/            # SMPL-H body model parameters
│   │   │   ├── body-pose/    # Body pose parameters
│   │   │   ├── global_orient/ # Global orientation parameters
│   │   │   ├── is_valid/     # Valid frames indicators
│   │   │   ├── left_hand_pose/ # Left hand pose parameters
│   │   │   ├── right_hand_pose/ # Right hand pose parameters
│   │   │   └── translation/  # Global translation parameters
│   │   ├── transcript/       # Time-aligned speech transcription
│   │   │   └── V<vendor>_S<session>_I<interaction>_P<participant>.jsonl
│   │   ├── vad/              # Voice activity detection
│   │   │   └── V<vendor>_S<session>_I<interaction>_P<participant>.jsonl
│   │   └── video/            # Raw HD video recordings
│   │       └── V<vendor>_S<session>_I<interaction>_P<participant>.mp4
│   ├── test/                 # Test split with similar structure
│   └── train/                # Training split with similar structure
└── naturalistic/             # Spontaneous conversations
    ├── dev/                  # Same structure as improvised/dev
    ├── test/                 # Same structure as improvised/test
    └── train/                # Same structure as improvised/train
```

Each file is named according to a consistent convention:
- `V<vendor_id>`: Collection site/vendor identifier
- `S<session_id>`: Unique session identifier
- `I<interaction_id>`: Specific interaction within a session
- `P<participant_id>`: Individual participant identifier

### Available Modalities and Features

Each interaction in the dataset includes:

| Modality | Description | File Format | Sample Rate | 
|----------|-------------|-------------|-------------|
| 🎥 Video | High-definition face-to-face footage | MP4 (H.264) | 30 FPS, 1080p |
| 🎙️ Audio | Denoised audio with separate channels | WAV | 48kHz, 16-bit |
| 📝 Transcript | Time-aligned speech transcription | JSONL | - |
| 🏃 SMPL-H | 3D body model parameters | NPY | 30 Hz |
| 🧠 Movement Features | Comprehensive quantified movement data | NPY | 30 Hz |
| 📊 Annotations | Human-annotated behavioral data | JSON | - |
| 🔊 VAD | Voice activity detection | JSONL | 100 Hz |
| 📦 Keypoints | Face and body keypoints | NPY | 30 Hz |

#### Annotation Types

The dataset includes several types of human annotations for rich behavioral analysis:

| Annotation | Description | Format |
|------------|-------------|--------|
| 1P-IS | xxx | JSON |
| 1P-R | xxx | JSON |
| 3P-IS | xxx | JSON |
| 3P-R | xxx | JSON |
| 3P-V | xxx | JSON |

#### Movement Feature Types

The movement directory contains rich behavioral features:

| Feature | Description | Format |
|---------|-------------|--------|
| emotion_arousal | Arousal intensity measurements | NPY |
| emotion_valence | Valence (positive/negative) measurements | NPY |
| emotion_scores | Detected emotion categorical scores | NPY |
| expression | Parametric facial expression encodings | NPY |
| FAUToken/FAUValue | Facial Action Unit tokens and intensity values | NPY |
| gaze_encodings | Neural encodings of gaze direction | NPY |
| head_encodings | Neural encodings of head position and rotation | NPY |
| frame_latent | Per-frame latent representations | NPY |
| alignment_head_rotation | Head rotation data for temporal alignment | NPY |
| alignment_translation | Translation parameters for temporal alignment | NPY |
| EmotionArousalToken/EmotionValenceToken | Discretized emotion tokens | NPY |
| hypernet_features | Features from hypernetwork processing | NPY |



### Dataset Versions

We provide multiple versions of the dataset to accommodate different research needs and computational constraints:

| Version | Size | Samples | Modalities | Description |
|---------|------|---------|------------|-------------|
| Mini | 10GB | 100 | All | Quick experimentation and API testing |
| Small | 100GB | 1,000 | All | Algorithm development and prototyping |
| Medium | 1TB | 10,000 | All | Serious model training and validation |
| Full | 50TB | 100,000+ | All | State-of-the-art research and benchmarking |

Sample datasets are available in the `/sample` directory with three pre-configured sizes:
- `/sample/10GB`: Mini version with essential samples
- `/sample/100GB`: Small version with broader coverage
- `/sample/1TB`: Medium version with comprehensive samples

You can download any version from HuggingFace or our public FAIR S3 bucket using:

```bash
# Download the full dataset (requires ~50TB storage)
seamless-interaction download --version full --output-dir /path/to/storage

# Or use the streaming API to access without downloading
seamless-interaction stream --version 10GB --cache-dir /tmp/seamless-cache
```

### Imitator Encoder Architecture

The Imitator Encoder is a transformer-based model that processes multimodal input sequences to generate unified representations of human interaction behaviors:

<div align="center">
<img src="https://balamuruganthambiraja.github.io/Imitator/media/img/teaser.png" alt="Imitator Encoder Architecture" width="600px">
</div>

The encoder processes audio, visual, and textual inputs through specialized encoders before fusing them in a joint multimodal transformer. This architecture enables:

- Cross-modal alignment of behavioral signals
- Temporal modeling of interaction dynamics
- Transfer learning between different interaction contexts
- Generation of coherent multimodal behavior

The Imitator model leverages our extensive movement features (FAU, emotion, gaze, etc.) alongside speech and language understanding to create comprehensive behavior models that can understand and predict natural human interactions.

#### File Format Specifications

Our data is stored in the following formats for optimal usability:

| Format | Description | Usage |
|--------|-------------|-------|
| NPY | NumPy array files | Efficient storage of numerical feature vectors, keypoints, and parameters |
| JSONL | JSON Lines | Time-aligned annotations with one event per line (e.g., transcripts, VAD) |
| JSON | JavaScript Object Notation | Structured metadata and annotations with timestamps |
| MP4 | MPEG-4 Part 14 | High-quality compressed video with H.264 encoding |
| WAV | Waveform Audio | Uncompressed audio for highest fidelity processing |

## 🧪 Research Applications

The Seamless Interaction Dataset enables research across multiple domains:

### Embodied AI and Virtual Agents
- Train agents that display natural nonverbal behaviors
- Model turn-taking dynamics and interaction rhythms
- Generate contextually appropriate responses to human behavior

### Multimodal Understanding
- Analyze cross-modal correlations between speech, gesture, and expressions
- Extract behavioral patterns from large-scale interaction data
- Develop models for understanding social dynamics

### Human-Computer Interaction
- Design interfaces that respond to subtle human cues
- Improve telepresence technologies with better behavioral modeling
- Create more natural conversational agents

### Animation and Content Creation
- Generate realistic human behaviors for animated characters
- Synthesize conversational dynamics for virtual production
- Create training data for digital human technologies

## 🛠️ Tools and Examples

We provide several tools to help researchers work with the dataset:

```python
# Extract behavioral features from raw data
from seamless_interaction.features import extract_features

# Generate synthetic behaviors with the Imitator model
from seamless_interaction.models import ImitatorModel

# Working with the dataset structure
from seamless_interaction import SeamlessDataset

# Load a specific interaction by ID
dataset = SeamlessDataset("path/to/data", split="train")
interaction = dataset.get_interaction("V03_S0846_I00000285")

# Access synchronized modalities
video = interaction.video
audio = interaction.audio
transcript = interaction.transcript
facial_features = interaction.movement.expression
emotions = interaction.movement.emotion_scores

```

Check out our [example notebooks](https://github.com/facebookresearch/seamless-interaction/tree/main/examples) for more detailed examples, including:

- Basic data loading and visualization
- Multimodal feature extraction
- Generating synthetic interactions with the Imitator model

## ⚠️ Known Limitations and Noise in Metadata

Due to the scale and complexity of collecting the Seamless Interaction dataset, there are several aspects that will be the focus of continued work and improvement in future versions:

### Errors in Human-Based Time-Stamping
The core unit of the dataset is an interaction. Interactions define *active time* in which participant conversation and behavior can be linked to a pair of prompts. We have observed instances of misaligned time-stamps in which:
- Annotated start/end times may be too early or too late
- Prompt text occasionally doesn't align with spoken material
- Ordering of prompts may contain off-by-one errors

These issues impact approximately 10% of interactions after our attempts at correction. We've made best efforts to automatically identify and rectify these errors.

### Time Stamping "Noise" in Moments of Interest (MOI)
While there's inherent subjectivity in defining an MOI, there are rare cases where:
- The described behavior represents only a subset of the observed behavior
- The duration of the MOI doesn't fully capture the annotated behavior

### Incorrect Assignment of Participant IDs
In rare cases, we've observed:
- Duplicate participant identifiers assigned to different people
- The same person mapped to different identifiers

### Unreleased "Meta Time"
Currently, the dataset only contains *active time* segments. The *meta time* between interactions (hundreds of hours of additional data) may be explored in future releases.

### Variation in Recording Site Consistency
This multi-site project shows variation in:
- Recording quality (speaker-bleed, participants staying in frame)
- Acting quality in *Improvised* segments
- Likelihood of time-stamping errors

All vendors met our technical requirements, but there is clear variation in production quality between sites.

## 🔍 Benchmarks

We provide standard benchmarks to enable consistent evaluation and comparison across different approaches using the Seamless Interaction Dataset:

| Benchmark | Description | Metrics | Baseline Score | Top Score |
|-----------|-------------|---------|---------------|-----------|
| Interaction Understanding | Predict interaction outcomes from partial observations | Accuracy, F1 | 0.78 / 0.76 | 0.89 / 0.85 |
| Behavior Generation | Generate plausible behavioral responses to stimuli | FID, User Study | 0.62 / 4.2/5 | 0.42 / 4.7/5 |
| Emotion Recognition | Classify emotional states from multimodal cues | Accuracy, mAP | 0.81 / 0.76 | 0.88 / 0.84 |
| Turn-Taking Prediction | Predict conversational turn transitions | Precision, Recall | 0.73 / 0.69 | 0.82 / 0.78 |
| Nonverbal Behavior Synthesis | Generate matching nonverbal behaviors for speech | FVD, User Preference | 0.58 / 68% | 0.39 / 86% |
| Cross-Cultural Understanding | Transfer learning across different cultural contexts | Accuracy, F1 | 0.72 / 0.68 | 0.81 / 0.77 |

Our benchmarks are designed to measure both algorithmic performance and human-evaluated quality. The baseline scores represent our initial models, while top scores show the current state-of-the-art on our leaderboard.

> [!NOTE] This is just a placeholder in case we release some benchmarks. We can replace this with the evaluation results we did.

## 🤝 Contributing

We welcome contributions from the research community! Here are some ways to contribute:

- **Bug Reports & Feature Requests**: Open issues on GitHub
- **Dataset Improvements**: Help enhance our preprocessing pipelines or annotations
- **Model Contributions**: Submit your models to our benchmarks
- **Documentation**: Improve our guides, tutorials, and API documentation
- **Sample Code**: Share example applications built with the dataset

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines, code of conduct, and submission processes.

## 📄 License & Data Usage Policy

The Seamless Interaction Dataset is licensed under CC-BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).

This means you are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes without explicit permission.


## 📑 Citation

If you use the Seamless Interaction Dataset in your research, please cite:

```bibtex
@inproceedings{seamless2025,
  title={Seamless Interaction: A Large-Scale Dataset of Human Interaction for Learning Social Dynamics},
  author={Seamless Next Team},
  year={2025},
}
```

## 🙏 Acknowledgments

This project was made possible thanks to contributions from:

- The thousands of participants who provided interaction data
- Our dedicated annotation team
- Research collaborators from multiple institutions
- FAIR (Facebook AI Research)
- The open-source community for valuable tools and libraries
- Our data collection partners across multiple sites
- Meta Reality Labs for supporting this research initiative
