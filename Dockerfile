# ========= Base with Python/FFmpeg (Debian, wheels เยอะกว่า) =========
FROM node:20-bookworm-slim AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ffmpeg curl ca-certificates git \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

# ========= Stage 1: Build (Node + Prisma) =========
FROM base AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY prisma ./prisma
RUN npx prisma generate

COPY requirements.txt ./
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
 
COPY python ./python
COPY . .
RUN npm run build

# ========= Stage 2: Runtime =========
FROM base AS runner
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/python ./python

# ✅ เพิ่มสองบรรทัดนี้
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN npm prune --omit=dev
EXPOSE 3000
CMD ["node", "dist/src/worker/main.js"]
