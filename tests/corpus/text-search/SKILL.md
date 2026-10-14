---
name: text-search
description: Find the files in a workspace that mention a phrase and explain each match.
allowed-tools: Read Grep Glob
---

# Text search

Locate where a phrase appears and say briefly why each match matters.

## Procedure

1. Search the workspace for the phrase with Grep.
2. For each matching file, return the path and a one-sentence explanation.
