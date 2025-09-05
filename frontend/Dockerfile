# Use the official Node.js image as a parent image
FROM node:20-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package.json ./
COPY package-lock.json* ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

# Expose the port the app runs on
EXPOSE 3000

# Start the Next.js development server
CMD ["npm", "run", "dev"]
