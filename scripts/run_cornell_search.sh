#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

chmod +x "$SCRIPT_DIR/param_search_cornell.sh"

timestamp=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$PROJECT_ROOT/logs/cornell_search"

cd "$PROJECT_ROOT"

nohup "$SCRIPT_DIR/param_search_cornell.sh" > "logs/cornell_search/search_${timestamp}.log" 2>&1 &

echo $! > cornell_search_pid.txt

echo "Cornell, Parameter search has been launched！"
