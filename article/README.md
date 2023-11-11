![tweet analyzer](https://github.com/snyamson/LP5-Twitter-Sentiment-Analysis/assets/58486437/fb4b413a-4c27-449d-bde7-2b3d2871b3cb)
# Analyzing COVID-19 Twitter Sentiments: A Deep Dive into NLP with Transformers

# Introduction
The COVID-19 pandemic has not only brought unprecedented health challenges but has also sparked a flood of public discourse on social media platforms, particularly Twitter. Leveraging the power of natural language processing (NLP) and deep learning models, the "COVID-19 Twitter Sentiment Analyzer" project seeks to systematically analyze and categorize sentiments expressed in a vast collection of COVID-19-related tweets. This initiative aims to provide valuable insights into public perceptions, concerns, and attitudes surrounding the pandemic.

# Business Understanding
Understanding public sentiment during a global health crisis is crucial for decision-makers, researchers, and organizations seeking to navigate the complex landscape of public opinion. The "COVID-19 Twitter Sentiment Analyzer" project aligns with this broader goal, aiming to enhance our understanding of public sentiment and foster responsible social media discourse.

# Importing Libraries
To kick off this venture, I began by importing essential libraries and installing the transformers library, a robust toolkit for working with pre-trained language models. These tools, ranging from pandas for data manipulation to torch for deep learning, set the stage for what would become a comprehensive sentiment analysis project.
```python
# Load necessary libraries
import os
import re
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Google drive
from google.colab import drive

# Deep learning
import torch
from torch import nn

# Scikit-Learn
from sklearn.model_selection import train_test_split

# Class imbalance
from sklearn.utils.class_weight import compute_class_weight

# Dataset preparation
from datasets import load_dataset, load_metric

# Transformers
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding

# Huggingface
from huggingface_hub import notebook_login

import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

```

These libraries lay the foundation for our project, providing tools for data manipulation, visualization, and access to state-of-the-art NLP models.
