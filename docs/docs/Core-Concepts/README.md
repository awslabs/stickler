---
title: Overview
---

# Core Concepts

This section covers the fundamental evaluation principles that power Stickler's comparison engine. Understanding these concepts is essential for effectively using the library to evaluate structured data.

## What's Covered

**Classification Logic** explains how Stickler categorizes comparisons into confusion matrix metrics (TP, FP, FN, TN, FD, FA). It defines the rules for classifying simple values, lists, and nested objects, including how null values and empty collections are handled.

**Hungarian Matching for Lists** describes the algorithm used to optimally pair elements when comparing lists of structured objects. This includes detailed examples of how similarity scores determine matches and how unmatched items are classified.

**Threshold-Gated Recursive Evaluation** introduces the core principle that governs when Stickler performs detailed nested field analysis. Only object pairs that meet the similarity threshold receive recursive evaluation, while poorly-matched pairs are treated atomically.

> **Note**
> These concepts work together to provide accurate, interpretable evaluation metrics. The classification logic defines what counts as a match, Hungarian matching determines optimal pairings, and threshold-gating controls the depth of analysis.

## Why These Concepts Matter

Stickler's evaluation approach differs from simple equality checks by considering similarity thresholds, optimal matching strategies, and hierarchical structure. These concepts ensure that evaluation metrics reflect meaningful differences rather than arbitrary misalignments, particularly when comparing lists where element order may vary.
