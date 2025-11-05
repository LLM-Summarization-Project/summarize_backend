# ---------- 1️⃣ Install dependencies (include devDeps for build)
FROM node:20-slim AS deps
WORKDIR /app

RUN corepack enable && corepack prepare pnpm@9 --activate
COPY package.json pnpm-lock.yaml* ./
# Include devDependencies so nest CLI is available
RUN pnpm install --no-frozen-lockfile

# ---------- 2️⃣ Build the app
FROM node:20-slim AS build
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@9 --activate

# Copy everything needed for build
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Run build (Nest CLI now available)
RUN pnpm build

# ---------- 3️⃣ Runtime (production only)
FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

# Copy only compiled code and prod dependencies
COPY --from=build /app/dist ./dist
COPY package.json pnpm-lock.yaml* ./

# Install only production dependencies
RUN corepack enable && corepack prepare pnpm@9 --activate && pnpm install --prod --frozen-lockfile

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD node -e "require('http').get('http://localhost:'+(process.env.PORT||3000)+'/health',res=>{if(res.statusCode!==200)process.exit(1)}).on('error',()=>process.exit(1))"
CMD ["node", "dist/main.js"]
