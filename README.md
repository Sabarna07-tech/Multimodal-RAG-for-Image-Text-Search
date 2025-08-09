# Multimodal RAG SaaS Platform

## Overview

This project provides a scalable, multi-tenant Retrieval-Augmented Generation (RAG) system designed as a foundation for a Software as a Service (SaaS) platform. It understands and processes both text and images from various sources (YouTube videos, PDF files), allowing users to chat with their private data.

The system is built with a modern Python stack, including FastAPI for the web framework, LangGraph for orchestrating AI logic, ChromaDB for vector storage, and Google's Gemini Pro Vision for multimodal generation.

---

## Architecture

The system is designed with a multi-tenant architecture, where each user's data is isolated and secured via API key authentication.

```
+---------------------+      +-------------------+      +----------------------+
|   User 1 (Web UI)   |----->|   API Gateway     |----->|   RAG Pipeline       |
| (API Key: key-1)    |      | (FastAPI)         |      | (User 1's Context)   |
+---------------------+      +-------------------+      +----------------------+
                                     |
+---------------------+              |
|   User 2 (Web UI)   |--------------+
| (API Key: key-2)    |
+---------------------+

+--------------------------+   +---------------------------+
| Vector Store (ChromaDB)  |   | Chat History (SQLite)     |
| - user-1_text_collection |   | - user-1_checkpoints.db   |
| - user-1_image_collection|   | - user-2_checkpoints.db   |
| - user-2_text_collection |   +---------------------------+
| - user-2_image_collection|
+--------------------------+
```

### Core Features:

- **Multi-Tenancy**: User data is isolated at both the vector store and chat history levels.
- **API Key Authentication**: Simple and effective security to control access.
- **Centralized Configuration**: Easy to manage settings for different environments.
- **Persistent Chat History**: Conversations are saved per-user, per-thread.
- **Modular and Scalable**: The project is structured to be easily extended with new features.

---

## Getting Started

### 1. Prerequisites

- Python 3.9+
- An environment variable manager (e.g., `direnv`) is recommended.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

### 3. Configuration

The application is configured using environment variables. Copy the example `.env` file and fill in your details.

```bash
cp .env.example .env
```

Now, edit the `.env` file:

- **`GOOGLE_API_KEY`**: You must provide a valid Google API key for the Gemini model to work.
- **`API_KEYS`**: This is a JSON string mapping API keys to user IDs. You can add your own keys. The default key is `test-key`.
  ```
  API_KEYS='{"test-key": "test-user", "my-secret-key": "user-alpha"}'
  ```

### 4. Running the Application

Once your `.env` file is configured, you can run the FastAPI server:

```bash
python main.py
```

The server will start on `http://localhost:8000`.

### 5. Using the Web Interface

Navigate to `http://localhost:8000` in your browser.

1.  **Enter Your API Key**: In the "API Key" section, enter one of the keys you defined in your `.env` file (e.g., `test-key`).
2.  **Add Data**: Use the forms to process a YouTube video or upload a PDF. This data will be added to the vector store associated with your API key.
3.  **Chat**: Ask questions about the data you've added. Your conversation will be saved and is unique to your user account.

---

## Project Structure

The project is organized into a main `app` package for better modularity:

- **`app/`**: The core application source code.
  - **`core/`**: Centralized configuration (`config.py`).
  - **`data_extraction/`**: Modules for extracting data from PDFs and YouTube.
  - **`embedding/`**: The text and image embedding module.
  - **`generation/`**: The Gemini generator module.
  - **`retrieval/`**: The retriever module.
  - **`vector_store/`**: The multi-tenant `ChromaStore` module.
  - **`static/`**: The HTML/JS frontend.
  - **`main.py`**: The FastAPI application logic.
- **`main.py`**: The root script to run the server.
- **`requirements.txt`**: Project dependencies.
- **`.env.example`**: An example configuration file.
- **`output/`**: Default directory for storing ChromaDB databases and chat histories. This directory is created automatically.
---
