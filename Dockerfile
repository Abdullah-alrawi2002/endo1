# All-in-one: Next.js web app + CT sidecar · one container · one public URL
# Deploy to Railway / Render / Fly / any Docker host.

# ---- CT Python deps ----
FROM python:3.11-bookworm AS ct-deps
WORKDIR /ct
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY ct-sidecar/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Node deps + Next build ----
FROM node:20-bookworm AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- Runtime ----
FROM python:3.11-bookworm AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    ENABLE_CT_MODULE=true \
    ENABLE_CORRECTION_RAG=true \
    CT_SIDECAR_URL=http://127.0.0.1:8000 \
    CT_ANALYSIS_STORE_PATH=/data/ct-analyses \
    CORRECTIONS_JSON_PATH=/data/corrections.json \
    nnUNet_raw=/models/raw \
    nnUNet_preprocessed=/models/preprocessed \
    nnUNet_results=/models/results \
    CT_ALLOWED_ORIGINS=*

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg supervisor \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ct-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=ct-deps /usr/local/bin /usr/local/bin

COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/lib/prompts ./lib/prompts
COPY --from=builder /app/next.config.ts ./next.config.ts
COPY ct-sidecar/app ./ct-sidecar/app

RUN mkdir -p /data/ct-analyses /models /var/log/supervisor \
    && chmod -R 777 /data /models /var/log/supervisor

COPY deploy/supervisord.conf /etc/supervisor/conf.d/endo.conf
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-3000}/api/features" || exit 1

CMD ["/entrypoint.sh"]
