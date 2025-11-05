# summarize_backend/Dockerfile
FROM node:20-slim AS deps
WORKDIR /app

# If you use PNPM (recommended)
RUN corepack enable && corepack prepare pnpm@9 --activate
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --no-frozen-lockfile

FROM node:20-slim AS build
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@9 --activate
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# If using Prisma:
# RUN pnpm prisma generate
RUN pnpm build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/dist ./dist
COPY --from=deps /app/node_modules ./node_modules
# HEALTHCHECK is optional; good to have:
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD node -e "require('http').get('http://localhost:'+(process.env.PORT||3000)+'/health',res=>{if(res.statusCode!==200)process.exit(1)}).on('error',()=>process.exit(1))"
# EXPOSE 3000
CMD ["node","dist/main.js"]
