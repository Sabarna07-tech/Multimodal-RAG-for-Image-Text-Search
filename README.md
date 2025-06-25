# Multimodal RAG for Image-Text Search

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that understands and processes both text and images to answer user queries. It can ingest data from various sources (e.g., YouTube videos, PDF files), extract relevant textual and visual information, and use that multimodal context to provide comprehensive answers.

---

## Architecture

The system is built on a modular pipeline, orchestrated by a `StateGraph` (from the `langgraph` library) that manages the flow of data between stages.

```
                           +-------------------+
                           |   User Interface  |
                           | (Web UI / Console)|
                           +-------------------+
                                     |
                                     v
+-------------------+      +-------------------+
|  Data Ingestion   |      |   User's Query    |
|(PDFs, YouTube, etc)+------>-------------------+
+-------------------+      |
         |                 v
         v       +---------------------+
+-----------------+  |  Embedding (CLIP)   |
|  Data Extraction|  +---------------------+
| (Text & Images) |            |
+-----------------+            v
         |           +---------------------+
         v           |      Retriever      |
+-----------------+  +---------------------+
| Embedding       |            |
| (Text & Images) |<-----------+
+-----------------+            |
         |                     v
         v       +--------------------------+
+-----------------+  | Multimodal LLM (Gemini)  |
| Vector Store    |  |  (Text + Image Context)  |
| (ChromaDB)      |  +--------------------------+
+-----------------+            |
                               v
                         +-----------------+
                         | Generated       |
                         | Answer          |
                         +-----------------+
```

---

## Core Components

### 1. Data Ingestion & Extraction

- **Sources**: Supports YouTube videos and PDF files.
- **YouTube**: Downloads video, extracts transcript with timestamps, samples frames at intervals.
- **PDF**: Extracts text and embedded images from each page.

### 2. Embedding

- **Purpose**: Converts both text and images into numerical vector representations (embeddings).
- **Text**: Uses `all-MiniLM-L6-v2` from `sentence-transformers`.
- **Image**: Uses `clip-ViT-B-32` (CLIP model) for semantic image embeddings.

### 3. Vector Store

- **Storage**: Uses `ChromaDB` to store embeddings.
- **Separation**: Maintains two collections—one for text (`text_embeddings`), one for images (`image_embeddings`).

### 4. Retrieval

- **Query Handling**: Embeds user's query using the text embedding model.
- **Search**: Finds semantically similar text and images in ChromaDB using the query embedding.

### 5. Generation

- **LLM**: Uses Google’s Gemini Pro Vision model (accepts both text and images).
- **Prompt Construction**: Combines the original query, retrieved text, and images.
- **Output**: Asks Gemini to generate a comprehensive answer utilizing all supplied information.

---

## How Multimodality is Achieved

1. **Dual Embedding Models**  
   Uses distinct embedding models for text and images, ensuring each modality is properly represented.

2. **CLIP for Cross-Modal Understanding**  
   Employs CLIP to create an embedding space where images and their text descriptions are close together, enabling image search via text queries.

3. **Separate Vector Collections**  
   Text and images are stored in distinct collections, allowing efficient, independent searching per modality before combining results.

4. **Multimodal LLM for Synthesis**  
   Gemini Pro Vision can reason over both text and image data, synthesizing answers that integrate information from both sources (e.g., analyzing a PDF chart and related text).

---

## Summary

This RAG system demonstrates a robust approach to multimodal retrieval and generation, leveraging state-of-the-art models and an efficient, modular architecture. By combining specialized embedding models, a powerful vector database, and a multimodal LLM, it delivers answers that draw from the full spectrum of available data—text and images alike.

---
