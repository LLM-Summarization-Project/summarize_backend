# ==========================================
# 🏗️ Stage 1: Build the NestJS application
# ==========================================
FROM node:20-alpine AS builder

# Create app directory
WORKDIR /app

# Copy package files first (for better caching)
COPY package*.json ./

# Install all dependencies (including dev)
RUN npm ci

# Copy the full source
COPY . .

# Build the NestJS app
RUN npm run build


# ==========================================
# 🚀 Stage 2: Run the production server
# ==========================================
FROM node:20-alpine AS runner

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/dist ./dist

# Install only production dependencies
RUN npm ci --omit=dev

# Expose the port that NestJS listens on
EXPOSE 3000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
  CMD wget --spider -q http://localhost:3000 || exit 1

# Start the app
CMD ["node", "dist/main.js"]
