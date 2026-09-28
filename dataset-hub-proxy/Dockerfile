FROM nginx:1.27-alpine

RUN apk add --no-cache apache2-utils

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /docker-entrypoint.d/40-basic-auth.sh

RUN chmod +x /docker-entrypoint.d/40-basic-auth.sh