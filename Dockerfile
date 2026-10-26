FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
WORKDIR /work
COPY . .
RUN python -m pip install --no-cache-dir '.[meshes]'
CMD ["python", "-c", "from barrel_mission import Mission; from barrel_environment import Scene; print('BARREL simulation framework ready')"]
