# Docker Setup Instructions

## Quick Start

### Using Docker Compose (Recommended)
```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Using Docker directly
```bash
# Build the image
docker build -t hackathon-scoring .

# Run the container
docker run -d \
  -p 6060:6060 \
  -v $(pwd)/scores.csv:/app/scores.csv \
  -v $(pwd)/state.json:/app/state.json \
  -v $(pwd)/config.json:/app/config.json \
  --name hackathon-scoring-app \
  hackathon-scoring

# View logs
docker logs -f hackathon-scoring-app

# Stop the container
docker stop hackathon-scoring-app
docker rm hackathon-scoring-app
```

## Access the Application

Once running, access the application at:
- **URL**: http://localhost:6060/thooral/scoring/login

## Data Persistence

The Docker setup mounts the following files as volumes to persist data:
- `scores.csv` - Stores all judge scores
- `state.json` - Stores application state (active round, scoring open/closed)
- `config.json` - Configuration (judges, teams, criteria)

Changes to these files will persist even if the container is restarted.

## Environment Variables

You can customize the following environment variables:
- `FLASK_ENV` - Set to `production` or `development` (default: production)

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Or for direct docker
docker logs hackathon-scoring-app
```

### Port already in use
If port 6060 is already in use, edit `docker-compose.yml` and change the port mapping:
```yaml
ports:
  - "8080:6060"  # Change 8080 to any available port
```

### Reset all data
```bash
# Stop and remove containers
docker-compose down

# Remove data files (WARNING: This deletes all scores!)
rm scores.csv state.json

# Start fresh
docker-compose up -d
```

## Development Mode

To run in development mode with hot reload:
```bash
docker run -d \
  -p 6060:6060 \
  -v $(pwd):/app \
  -e FLASK_ENV=development \
  --name hackathon-scoring-dev \
  hackathon-scoring \
  python app.py
```
