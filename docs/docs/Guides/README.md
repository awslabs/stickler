---
title: Overview
---

# Guides

This section provides practical guides for working with Stickler's StructuredModel comparison system. These guides cover common use cases and advanced features with working examples.

## What's Covered

**StructuredModel compare_with Method** walks through how the core comparison method works, from basic usage to understanding the internal flow. It explains the recursive traversal process, field-by-field analysis, and how results are assembled.

**Universal Aggregate Field Feature** describes the automatic aggregation of confusion matrix metrics at every level of the comparison tree. This feature provides field-level granularity without requiring manual configuration.

**StructuredModel Advanced Functionality** provides a technical deep-dive into the internal comparison engine. It covers the recursive logic, field dispatch system, Hungarian matching integration, and score aggregation mechanisms for developers who need to extend or debug the system.

**StructuredModel Dynamic Creation** explains how to create StructuredModel classes from JSON configuration. This enables configuration-driven model definitions with full comparison capabilities, including nested models and custom comparators.

## How to Use These Guides

Start with the compare_with guide to understand basic usage and the comparison flow. The Universal Aggregate Field guide explains how to access aggregated metrics in your results. For advanced scenarios or customization, consult the Advanced Functionality guide. If you need to generate models programmatically, the Dynamic Creation guide shows how to define models using JSON configuration.

Each guide includes code examples and practical scenarios to help you apply the concepts to your own use cases.
