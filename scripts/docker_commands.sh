#!/bin/bash
# Helper script for managing Docker environment for HADES experiments

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function print_help {
    echo -e "${YELLOW}HADES Docker Environment Management${NC}"
    echo ""
    echo "Usage: ./scripts/docker_commands.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build          Build Docker images"
    echo "  up             Start all containers"
    echo "  down           Stop and remove all containers"
    echo "  restart        Restart all containers"
    echo "  shell          Open shell in the hades container"
    echo "  logs           View logs from all containers"
    echo "  status         Check status of containers"
    echo "  pull-models    Pull required Ollama models (qwen:2.5-coder)"
    echo "  run-tests      Run test suite inside container"
    echo "  experiment     Run an experiment (requires experiment name)"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/docker_commands.sh build"
    echo "  ./scripts/docker_commands.sh experiment phase1-pathrag"
}

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

case "$1" in
    build)
        echo -e "${GREEN}Building Docker images...${NC}"
        docker-compose build
        ;;
    up)
        echo -e "${GREEN}Starting containers...${NC}"
        docker-compose up -d
        echo -e "${GREEN}Containers are now running in the background${NC}"
        ;;
    down)
        echo -e "${YELLOW}Stopping and removing containers...${NC}"
        docker-compose down
        ;;
    restart)
        echo -e "${YELLOW}Restarting containers...${NC}"
        docker-compose restart
        ;;
    shell)
        echo -e "${GREEN}Opening shell in hades container...${NC}"
        docker-compose exec hades bash
        ;;
    logs)
        echo -e "${GREEN}Viewing logs (Ctrl+C to exit)...${NC}"
        docker-compose logs -f
        ;;
    status)
        echo -e "${GREEN}Container status:${NC}"
        docker-compose ps
        ;;
    pull-models)
        echo -e "${GREEN}Pulling required Ollama models...${NC}"
        # Wait for Ollama to be ready
        echo "Waiting for Ollama service to be ready..."
        sleep 5
        docker-compose exec ollama ollama pull qwen:2.5-coder
        echo -e "${GREEN}Models pulled successfully${NC}"
        ;;
    run-tests)
        echo -e "${GREEN}Running test suite...${NC}"
        docker-compose exec hades pytest -v
        ;;
    experiment)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Experiment name is required${NC}"
            echo "Usage: ./scripts/docker_commands.sh experiment [experiment-name]"
            exit 1
        fi
        echo -e "${GREEN}Running experiment: $2${NC}"
        docker-compose exec hades python -m experiments.run --experiment "$2"
        ;;
    help|*)
        print_help
        ;;
esac
