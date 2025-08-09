# --- Base Image ---
# Use a slim Python image for a smaller final image size.
FROM python:3.12-slim

# --- Environment Variables ---
# Prevents Python from writing pyc files to disc.
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to the terminal without buffering.
ENV PYTHONUNBUFFERED 1

# --- System Dependencies ---
# Install any system-level dependencies if needed.
# For now, none are required, but this is where you would add them.
# RUN apt-get update && apt-get install -y ...

# --- Working Directory ---
# Set the working directory inside the container.
WORKDIR /app

# --- Install Dependencies ---
# Copy the requirements file first to leverage Docker's layer caching.
# The layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# Install dependencies, using the same CPU-only torch strategy to keep the image small.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
# Copy the rest of the application code into the working directory.
COPY ./app ./app
COPY ./main.py .

# --- Expose Port ---
# Expose the port the application will run on.
EXPOSE 8000

# --- Run Command ---
# The command to run when the container starts.
# We run uvicorn directly, binding it to all network interfaces.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
