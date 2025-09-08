#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

chmod +x "$SCRIPT_DIR/param_search_actor_frame.sh"

timestamp=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$PROJECT_ROOT/logs/actor_frame"

cd "$PROJECT_ROOT"

nohup "$SCRIPT_DIR/param_search_actor_frame.sh" > "logs/actor_frame/search_${timestamp}.log" 2>&1 &

echo $! > actor_frame_search_pid.txt

echo "Actor frame method, Parameter search has been launched！"