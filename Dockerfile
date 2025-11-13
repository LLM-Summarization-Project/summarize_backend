# ใช้ Python base ที่มี venv + requirements ติดตั้งไว้แล้ว
# จะใช้ tag อะไรก็ได้ ตามที่คุณ build ไว้จากไฟล์แรก
FROM northpat/summary-python-base:1 AS base

# ========= Stage 1: Build (Node + Prisma + Nest build) =========
FROM base AS builder
WORKDIR /app

# ติดตั้ง dependency ของ Node
COPY package*.json ./
RUN npm ci

# Prisma
COPY prisma ./prisma
RUN npx prisma generate

# โค้ดแอป + python scripts
COPY python ./python
COPY . .

# build NestJS
RUN npm run build

# ========= Stage 2: Runtime =========
FROM base AS runner
WORKDIR /app

# ดึงไฟล์ที่ build แล้วเข้ามา
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/python ./python

# base มี /opt/venv + PATH แล้ว ไม่ต้อง COPY /opt/venv อีก
# ถ้าอยาก prune dev deps ของ Node
RUN npm prune --omit=dev

EXPOSE 3000
CMD ["node", "dist/src/worker/main.js"]