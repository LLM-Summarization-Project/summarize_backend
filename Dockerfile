# ========= Stage 1: Build Node.js (ใช้ official node image) =========
FROM node:20-slim AS node-builder
WORKDIR /app

# ติดตั้ง dependency ของ Node
COPY package*.json ./
RUN npm ci

# Prisma
COPY prisma ./prisma
RUN npx prisma generate

# โค้ดแอป
COPY . .

# build NestJS
RUN npm run build

# ========= Stage 2: Runtime (Python base + Node.js runtime) =========
FROM northpat/summary-python-base:1 AS runner
WORKDIR /app

# ติดตั้ง Node.js runtime
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# ดึงไฟล์ที่ build แล้วเข้ามา
COPY --from=node-builder /app/dist ./dist
COPY --from=node-builder /app/node_modules ./node_modules
COPY --from=node-builder /app/package*.json ./
COPY --from=node-builder /app/prisma ./prisma

# Re-generate Prisma client for the correct runtime platform (OpenSSL version)
RUN npx prisma generate

# Python scripts
COPY python ./python

# Prune dev deps
RUN npm prune --omit=dev

EXPOSE 3000
CMD ["node", "dist/src/worker/main.js"]