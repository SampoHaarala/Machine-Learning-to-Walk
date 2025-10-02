#!/bin/bash
#
# Launch both the Python SNN server and the Unity simulation.
#
# This script first starts the Python server in the background and
# waits a short period to ensure the server is listening before
# launching Unity.  Replace the path to the Unity executable with
# your own build.  To stop both processes press Ctrl+C.

PACKAGE_DIR="$(dirname "$0")/.."

# Use the module invocation to ensure Python can locate the
# ``snn_unity_project`` package regardless of the working directory.
PYTHON_CMD=(python3 -m snn_unity_project.python.main)
UNITY_EXEC="./YourUnityGame.x86_64"  # TODO: update this to your Unity build

echo "[run_all] Starting Python SNN server…"
"${PYTHON_CMD[@]}" --host 127.0.0.1 --port 9000 &
PYTHON_PID=$!

# Allow the server a moment to start listening
sleep 2

# Launch the Unity executable.  You can pass additional arguments
if [ -x "$UNITY_EXEC" ]; then
  echo "[run_all] Launching Unity simulation…"
  "$UNITY_EXEC" &
  UNITY_PID=$!
else
  echo "[run_all] Warning: Unity executable '$UNITY_EXEC' not found or not executable"
fi

# Forward Ctrl+C to both processes
trap 'echo "[run_all] Stopping…"; kill $PYTHON_PID $UNITY_PID 2>/dev/null' INT

# Wait for both processes to finish
wait $PYTHON_PID $UNITY_PID

echo "[run_all] Both processes exited"
