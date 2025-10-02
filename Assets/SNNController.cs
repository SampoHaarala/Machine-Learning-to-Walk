using System;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using UnityEngine;

/*
 * SNNController
 *
 * Attach this component to a GameObject in your Unity scene to
 * control the character via a spiking neural network running in
 * Python.  The component maintains a TCP connection to the Python
 * server, sends features each physics tick and applies the
 * returned actions to the character joints.
 *
 * The script assumes you have a humanoid with configurable joints
 * representing hips, knees, ankles, spine, shoulders and elbows.
 * You must populate the `joints` array in the Inspector with the
 * appropriate Joint components (e.g. ArticulationBody or
 * ConfigurableJoint) in the same order as the Python model's
 * outputs.
 */

public class SNNController : MonoBehaviour
{
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
    public int port = 6969;

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
    private StreamReader _reader;
    private StreamWriter _writer;
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
            _reader = new StreamReader(_stream);
            _writer = new StreamWriter(_stream) { NewLine = "\n", AutoFlush = true };
            Debug.Log($"[SNNController] Connected to SNN server at {host}:{port}");

            // ---- Handshake receive ----
            string line = _reader.ReadLine();
            if (string.IsNullOrEmpty(line))
            {
                Debug.LogError("[SNNController] Handshake failed: no data"); return;
            }

            try
            {
                string[] parts = line.Split(',');
                if (parts.Length >= 4)
                {
                    // parse directly into fields
                    this.rewardsSize = int.Parse(parts[0]);
                    this.patienceMax = int.Parse(parts[1]);
                    this.dt = float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture);
                    Debug.Log($"[SNNController] Handshake OK");
                }
                else
                {
                    Debug.LogError("[SNNController] Handshake malformed: " + line);
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError("[SNNController] Handshake parse error: " + ex.Message);
            }
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
            // Get all angular velocity data of the joint.
            Vector3 wLocal = cj.transform.InverseTransformDirection(rb.angularVelocity) * Mathf.Rad2Deg;
            if (IsAxisLocked(0))
            {
                features[i] = wLocal.x;
                i++;
            }
            if (IsAxisLocked(1))
            {
                features[i] = wLocal.y;
                i++;
            }
            if (IsAxisLocked(2))
            {
                features[i] = wLocal.z;
            }
        }

        // Reward is received from movement in the z-axis.
        // The reward is the dot product of the movement of the physics body
        // and the z-axis base vector.
        float currentVelocity = Vector3.Dot(rb.velocity, Vector3.forward);
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

        // Send data to SNN
        try
        {
            _writer.Write(sb.ToString());
        }
        catch (Exception ex)
        {
            Debug.LogError($"[SNNController] Communication error trying to send data: {ex.Message}");
            _connected = false;
        }

        // Receive command from SNN
        if (_stream.DataAvailable)
        {
            string line = _reader.ReadLine();
            if (!string.IsNullOrEmpty(line))
            {
                string[] parts = line.Split(',');
                int actionCount = Math.Min(parts.Length, joints.Length);
                for (int i = 0; i < actionCount; i++)
                {
                    if (float.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out float action))
                    {
                        ApplyActionToJoint(joints[i], action);
                    }
                }
            }
        }

        // Update simulation once communication is done.
        Physics.Simulate(dt);
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

    /// <summary>
    /// Check if axis is locked. 
    /// </summary>
    /// <param name="j">The joint component to check.</param>
    /// <param name="axis">The axis to be checked.</param>
    bool IsAxisLocked(ConfigurableJoint j, int axis)
    {
        switch (axis)
        {
            case 0: return Mathf.Approximately(j.lowTwistLimit.limit, 0f) && Mathf.Approximately(j.highTwistLimit.limit, 0f);
            case 1: return Mathf.Approximately(j.swing1Limit.limit, 0f);
            case 2: return Mathf.Approximately(j.swing2Limit.limit, 0f);
            default: return true;
        }
    }
}
