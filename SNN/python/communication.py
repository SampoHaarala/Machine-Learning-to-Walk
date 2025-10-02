"""Communication module for the Unity–Python bridge.

This module defines a simple TCP server that listens for incoming
connections from a Unity client.  The server expects each message
to be a newline‑terminated string of comma‑separated floats.  The
server passes the list of floats to a user‑defined callback and
returns the callback's result as another comma‑separated float list.

Example usage::

    from snn_unity_project.python.snn_model import SpikingNetwork
    from snn_unity_project.python.communication import UnityCommunicationServer

    network = SpikingNetwork(input_size=…, output_size=…)
    def handle_obs(obs):
        actions = network.step(obs)
        return actions
    server = UnityCommunicationServer(host='127.0.0.1', port=9000)
    server.start(handle_obs)

The protocol is deliberately simple to make debugging easier. 
"""

from __future__ import annotations

import socket, threading, json, time
from typing import Callable, Iterable, List


class UnityCommunicationServer:
    """A lightweight TCP server for communicating with Unity.

    The server accepts a single client connection and processes lines
    of comma‑separated floats.  For each line received it calls
    ``callback`` with the list of floats and sends the return value
    (another list of floats) back to the client.  Messages are
    newline‑terminated to allow streaming.

    Parameters
    ----------
    host: str
        IP address to bind the server on.  Use ``'127.0.0.1'`` to
        restrict connections to the local machine.
    port: int
        TCP port to listen on.
    backlog: int
        Maximum number of queued connections.  Default is 1 as
        typically only one Unity client connects at a time.
    ````
    """

    def __init__(self, dt, rewards_size, patience_max, host: str = "127.0.0.1", port: int = 9000, backlog: int = 1) -> None:
        self.dt = dt
        self.rewards_size = rewards_size
        self.patience_max = patience_max
        self.host = host
        self.port = port
        self.backlog = backlog
        self._server: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self, callback: Callable[[List[float]], List[float]]) -> None:
        """Start the server and begin processing connections.

        This method blocks until the server is stopped.  Internally
        it spawns a thread to accept connections and process them.

        Parameters
        ----------
        callback: callable
            Function called with the list of floats decoded from each
            incoming message.  It should return a list of floats of
            the same length as the expected action vector.
        """
        if self._running:
            raise RuntimeError("Server already running")
        self._running = True
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(self.backlog)
        print(f"[Communication] Listening on {self.host}:{self.port}…")

        def client_thread(conn: socket.socket, addr: tuple[str, int]) -> None:
            """Handle communication with a single Unity client.
            Reads lines terminated by line-ends, decodes them into floats,
            calls the user callback and sends back the returned floats.
            """
            buffer = ""
            with conn:
                print(f"[Communication] Connected by {addr}")

                 # ---- Handshake ----
                handshake = f"{self.rewards_size},{self.patience_max},{self.dt}\n"
                try:
                    conn.sendall(handshake.encode("utf-8"))
                    print(f"[Communication] Sent handshake: {handshake.strip()}")
                except BrokenPipeError:
                    print("[Communication] Failed to send handshake")
                    return

                # ---- Normal looping ----
                while self._running:
                    try:
                        data = conn.recv(4096)
                    except ConnectionResetError:
                        print("[Communication] Connection reset by peer")
                        break
                    if not data:
                        break
                    buffer += data.decode("utf-8")
                    # process all complete lines in the buffer
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # decode comma‑separated floats
                            values = [float(x) for x in line.split(',') if x]
                        except ValueError:
                            print(f"[Communication] Could not parse floats from line: {line}")
                            continue
                        try:
                            result = callback(values)
                        except Exception as exc:
                            print(f"[Communication] Callback error: {exc}")
                            result = []
                        # encode result back to string
                        response = ",".join(f"{v:.6f}" for v in result) + "\n"
                        try:
                            conn.sendall(response.encode("utf-8"))
                        except BrokenPipeError:
                            print("[Communication] Broken pipe when sending response")
                            self._running = False
                            break
                print(f"[Communication] Client {addr} disconnected")

        def accept_loop() -> None:
            while self._running:
                try:
                    conn, addr = self._server.accept()
                except OSError:
                    break
                thread = threading.Thread(target=client_thread, args=(conn, addr), daemon=True)
                thread.start()
        # Accept connections in this thread (blocking)
        try:
            accept_loop()
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the server and close any open sockets."""
        if not self._running:
            return
        self._running = False
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
        print("[Communication] Server stopped")