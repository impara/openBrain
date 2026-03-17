FROM node:20-alpine AS builder

WORKDIR /app

COPY crm_ui/package.json crm_ui/next.config.mjs /app/
COPY crm_ui/app /app/app
COPY crm_ui/public /app/public

RUN npm install --production=false && npm run build

FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/.next/standalone /app/
COPY --from=builder /app/.next/static /app/.next/static
COPY --from=builder /app/public /app/public

EXPOSE 3000

CMD ["node", "server.js"]

