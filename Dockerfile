# --- Build Stage ---
FROM python:3.9-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including 'git'
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libfontconfig1 \
    shared-mime-info && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies into a "wheelhouse"
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt


# --- Final Stage ---
FROM python:3.9-slim

# Create a non-root user and group for security
RUN addgroup --system app && adduser --system --group app

# Install only the runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libfontconfig1 \
    shared-mime-info && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the pre-built Python packages from the builder stage
COPY --from=builder /wheels /wheels
# Install the packages from the local wheels
RUN pip install --no-cache /wheels/*

# Copy your application code into the container
COPY . .

# Create the export directory and give ownership to the 'app' user
RUN mkdir -p /app/export/pdfs && \
    chown -R app:app /app

# Switch to the non-root user
USER app

# Expose port 5000 to the outside world
EXPOSE 5000

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run_server:app"]