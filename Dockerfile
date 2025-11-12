# ========= Base with Python/FFmpeg =========
FROM node:20-bookworm-slim AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ffmpeg curl ca-certificates git \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

# ========= Stage 1: Build (Node + Prisma) =========
FROM base AS builder
WORKDIR /app

COPY package.json pnpm-lock.yaml ./
RUN corepack enable && corepack prepare pnpm@9 --activate && pnpm install --frozen-lockfile

COPY prisma ./prisma
RUN npx prisma generate

COPY requirements.txt ./
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY python ./python
COPY . .
RUN pnpm build

# ========= Stage 2: Runtime =========
FROM base AS runner
WORKDIR /app

# Copy necessary build artifacts from the builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/python ./python
COPY --from=builder /opt/venv /opt/venv

# ✅ Add pnpm in the runtime stage (corepack not inherited)
RUN corepack enable && corepack prepare pnpm@9 --activate

# ✅ Prune dev dependencies
RUN pnpm prune --prod

# ✅ Make Python virtual env available
ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 3000
CMD ["node", "dist/src/worker/summarize.worker.js"]

