# Use the official Python image as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the backend requirements file into the container
COPY backend/requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code into the container
COPY backend/ ./

# Expose the port the app runs on (adjust if needed)
EXPOSE 8000

# Set the default command to run the backend (adjust if needed)
CMD ["python", "uvicorn", "app:app"]
