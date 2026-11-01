FROM python:3.12-slim-bookworm

ARG GARAK_REVISION

RUN test -n "${GARAK_REVISION}" \
    && apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/garak \
    && git -C /opt/garak remote add origin https://github.com/NVIDIA/garak.git \
    && git -C /opt/garak fetch --depth 1 origin "${GARAK_REVISION}" \
    && git -C /opt/garak checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/garak rev-parse HEAD)" = "${GARAK_REVISION}"

RUN python -m pip install --no-cache-dir /opt/garak

COPY docker/garak/patch_reasoning_content.py /tmp/patch_reasoning_content.py
RUN python /tmp/patch_reasoning_content.py && rm /tmp/patch_reasoning_content.py

WORKDIR /opt/garak
ENTRYPOINT ["python", "-m", "garak"]
