using System;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using UnityEngine;

/*
 * SNNController
 *
 * Attach this component to a GameObject in a Unity scene to
 * control the character via a spiking neural network running in
 * Python.  The component maintains a TCP connection to the Python
 * server, sends features each physics tick and applies the
 * returned actions to the character joints.
 *
 * The script assumes you have a humanoid with configurable joints
 * representing hips, knees, ankles, spine, shoulders and elbows.
 * You must populate the `joints` array in the Inspector with the
 * appropriate CharacterJoint components in the same order as the 
 * Python model's outputs.
 */

public class SNNController : MonoBehaviour
{
    // Reward are received from the acceleration towards the z-axis.
    private float previousZ = 0;

    // Store a rolling window of recent episode rewards.
    // A larger window smooths out noise when checking for progress.
    private int rewardsSize = 10;
    private Queue<float> rewards = new Queue<float>();

    // Track how many consecutive checks show no meaningful improvement.
    // If this counter exceeds patienceMax, we treat it as a plateau
    // and increase exploration noise to escape it.
    private int patienceMax = 10;
    private int patience = 0;

    [Header("Network settings")]
    public string host = "127.0.0.1";
    public int port = 9000;

    [Header("Joint configuration")]
    // List of joints controlled by the neural network.  The size of
    // this array should match the output dimensionality of the Python
    // model (i.e. number of degrees of freedom).  You can assign
    // joint components here via the Inspector.
    public CharacterJoint[] joints;

    // Rigidbody for calculating velocity.
    public Rigidbody rb;

    // Internal TCP client and stream
    private TcpClient _client;
    private NetworkStream _stream;
    private byte[] _readBuffer = new byte[4096];
    private StringBuilder _recvBuilder = new StringBuilder();

    // Simulation timing
    private bool _connected = false;
    private float dt = 0.02f;

    async void Start()
    {
        // Stop automatic simulation. This way simulation speed can be customised.
        //Physics.autoSimulation = false;
        // Get the joints.
        joints = GetComponentsInChildren<CharacterJoint>();

        // For debugging.
        foreach (CharacterJoint joint in joints)
        {
            Debug.Log("Found joint: " + joint.name);
        }
        // Connect to the Python server asynchronously
        try
        {
            _client = new TcpClient();
            await _client.ConnectAsync(host, port);
            _stream = _client.GetStream();
            _connected = true;
            Debug.Log($"[SNNController] Connected to SNN server at {host}:{port}");

            // TO-DO Retrieve simulation settings from handshake
            // and set all variable values.
            // rewardsSize = ...
            // patienceMax = ...
            // dt = ...
        }
        catch (Exception ex)
        {
            Debug.LogError($"[SNNController] Could not connect to SNN server: {ex.Message}");
        }
    }

    void OnDestroy()
    {
        // Clean up the connection when the object is destroyed
        if (_stream != null)
            _stream.Close();
        if (_client != null)
            _client.Close();
    }

    void FixedUpdate()
    {
        if (!_connected || joints == null || joints.Length == 0)
            return;

        // Get all features from the simulation.
        float[] features = new float[21];
        for (int i = 0; i < features.Length; i++)
        {
            // TO-DO Retrieve rotation of all the axis
            // allowed to move in each joint.
            features[i] = 0f;
        }

        // Reward is received from movement in the z-axis.
        // The reward is the dot product of the movement in z-
        float currentVelocity = this.gameObject.transform.forward.normalized.z;
        float reward = 0;
        if (rewards.Count < rewardsSize)
        {
            reward = currentVelocity;
            rewards.Enqueue(reward);
        }
        else
        {
            reward = currentVelocity;
            rewards.Enqueue(reward);
            rewards.Dequeue();
        }

        // Encode features and reward into a comma‑separated string
        // terminated by a newline. The Python server will parse this
        // message into floats.
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < features.Length; i++)
        {
            sb.Append(features[i].ToString("F6"));
            sb.Append(',');
        }
        sb.Append(reward.ToString("F6"));
        sb.Append('\n');
        byte[] sendBytes = Encoding.ASCII.GetBytes(sb.ToString());

        try
        {
            // Send data to Python
            _stream.Write(sendBytes, 0, sendBytes.Length);

            // Read response. We accumulate data until a newline is found.
            int bytesRead = _stream.Read(_readBuffer, 0, _readBuffer.Length);
            if (bytesRead > 0)
            {
                _recvBuilder.Append(Encoding.ASCII.GetString(_readBuffer, 0, bytesRead));
                int newlineIndex = _recvBuilder.ToString().IndexOf('\n');
                if (newlineIndex >= 0)
                {
                    string line = _recvBuilder.ToString(0, newlineIndex);
                    _recvBuilder.Remove(0, newlineIndex + 1);
                    // Parse returned actions
                    string[] parts = line.Split(',');
                    int actionCount = Math.Min(parts.Length, joints.Length);
                    for (int i = 0; i < actionCount; i++)
                    {
                        if (float.TryParse(parts[i], out float action))
                        {
                            ApplyActionToJoint(joints[i], action);
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[SNNController] Communication error: {ex.Message}");
            _connected = false;
        }

        // Update simulation once communication is done.
        //Physics.Simulate(dt);
    }

    /// <summary>
    /// Apply a scalar action value to a joint.  This method
    /// sets joint motor targets based on the network's output.  
    /// </summary>
    /// <param name="joint">The joint component to control.</param>
    /// <param name="action">The network's output for this joint, typically in [-1, 1].</param>
    private void ApplyActionToJoint(CharacterJoint joint, float action)
    {
        // TO-DO Implement
    }
}


