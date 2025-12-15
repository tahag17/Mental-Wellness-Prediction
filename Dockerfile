# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY pyapi/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Expose the port
EXPOSE 8080

# Run the app using the entrypoint
CMD ["python", "pyapi/main.py"]
