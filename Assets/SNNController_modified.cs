using System;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

/*
 * SNNController_modified
 *
 * This simplified controller talks to a generic neural network server
 * (MLP or CNN) to drive a character without physics. Instead of
 * computing rewards and applying torques, it directly reads the
 * current joint orientations, sends them over TCP and applies the
 * predicted orientations returned by the network.  It assumes each
 * joint has up to three degrees of freedom (X,Y,Z) and uses local
 * Euler angles in degrees as both inputs and outputs.  Locked axes
 * are still included in the feature vector, but you may choose to
 * ignore them in the network.
 *
 * Attach this component to your character and populate the
 * `joints` array via the Inspector.  The order of joints and axes
 * must match the order used when training your model.
 */
public class SNNControllerModified : MonoBehaviour
{
    [Header("Network settings")]
    public string host = "127.0.0.1";
    public int port = 6900;
    // dt in seconds (used only for handshake)
    [SerializeField] private float dt = 0.02f;

    [Header("Joint configuration")]
    public Transform[] joints;

    [System.Serializable]
    public struct AxisLimit {
        [Tooltip("min/max in wrapped degrees [-180,180]")] public float min, max;
    }

    [System.Serializable]
    public struct JointLimits {
        public string name;        // for inspector clarity
        public AxisLimit X, Y, Z;  // per-axis limits (wrapped degrees)
    }
    
    [Header("Per-joint angle limits (same length/order as 'joints')")]
    private JointLimits[] jointLimits;       // will be auto-filled by the menu action
    [Tooltip("Max degrees/sec per axis to smooth motion (0 = off)")]
    public float slewDegreesPerSec = 0f;

    // internal TCP client and stream
    private TcpClient _client;
    private NetworkStream _stream;
    private StreamReader _reader;
    private StreamWriter _writer;
    private bool _connected = false;
    private bool _awaiting = false;
    private float[] pendingAction = null;
    private volatile bool awaitingAction = false;

    static float Wrap180(float a) => Mathf.Repeat(a + 180f, 360f) - 180f;
    static float Clamp180(float a, float min, float max) => Mathf.Clamp(Wrap180(a), min, max);

    // --- Call this once to populate recommended human-ish limits based on joint names ---
    [ContextMenu("Apply Recommended Limits")]
    public void ApplyRecommendedLimits() {
        if (joints == null) return;
        jointLimits = new JointLimits[joints.Length];

        for (int i = 0; i < joints.Length; i++) {
            var t = joints[i];
            var name = t != null ? t.name : $"Joint_{i}";
            var lim = new JointLimits { name = name };

            string n = name.ToLowerInvariant();

            if (n.Contains("hip")) {
                // Hip: X [-45, 90], Y [-45, 45], Z [-30, 30]
                lim.X = new AxisLimit { min = -45, max =  90 };
                lim.Y = new AxisLimit { min = -45, max =  45 };
                lim.Z = new AxisLimit { min = -30, max =  30 };
            }
            else if (n.Contains("knee")) {
                // Knee: flexion only; lock Y/Z
                lim.X = new AxisLimit { min =   0, max = 140 };
                lim.Y = new AxisLimit { min =   0, max =   0 };
                lim.Z = new AxisLimit { min =   0, max =   0 };
            }
            else if (n.Contains("ankle") || n.Contains("foot")) {
                // Ankle
                lim.X = new AxisLimit { min = -30, max =  30 };
                lim.Y = new AxisLimit { min = -15, max =  15 };
                lim.Z = new AxisLimit { min = -15, max =  15 };
            }
            else if (n.Contains("spine") || n.Contains("chest") || n.Contains("back")) {
                // Spine
                lim.X = new AxisLimit { min = -30, max =  30 };
                lim.Y = new AxisLimit { min = -45, max =  45 };
                lim.Z = new AxisLimit { min = -30, max =  30 };
            }
            else if (n.Contains("shoulder") || n.Contains("clavicle")) {
                // Shoulder (wide)
                lim.X = new AxisLimit { min = -90, max =  90 };
                lim.Y = new AxisLimit { min = -90, max =  90 };
                lim.Z = new AxisLimit { min = -90, max =  90 };
            }
            else if (n.Contains("elbow") || n.Contains("arm")) {
                // Elbow: flexion only; lock Y/Z
                lim.X = new AxisLimit { min =   0, max = 145 };
                lim.Y = new AxisLimit { min =   0, max =   0 };
                lim.Z = new AxisLimit { min =   0, max =   0 };
            }
            else {
                // Fallback conservative clamp
                lim.X = new AxisLimit { min = -30, max = 30 };
                lim.Y = new AxisLimit { min = -30, max = 30 };
                lim.Z = new AxisLimit { min = -30, max = 30 };
            }

            jointLimits[i] = lim;
        }

    #if UNITY_EDITOR
        UnityEditor.EditorUtility.SetDirty(this);
    #endif
        Debug.Log("[Limits] Applied recommended per-joint limits based on names.");
    }

    void Awake()
    {
        // Connect to the Python server on start
        _ = ConnectAndHandshakeAsync();
        ApplyRecommendedLimits();
    }

    void OnDestroy()
    {
        if (_stream != null) _stream.Close();
        if (_client != null) _client.Close();
    }

    void Update()
    {
        if (!_connected || _awaiting) return;
        // kick one send->recv round; we don't await to avoid blocking Update,
        // but _awaiting prevents overlap until this finishes.

        if (!_connected) return;
        // If a prediction arrived, apply it
        if (pendingAction != null)
        {
            ApplyAction(pendingAction);
            pendingAction = null;
        }
        // Request next prediction
        RequestActionAsync();
    }

    private async Task ConnectAndHandshakeAsync()
    {
        try
        {
            _client = new TcpClient(AddressFamily.InterNetwork);
            await _client.ConnectAsync(IPAddress.Loopback, port);
            _stream = _client.GetStream();
            _reader = new StreamReader(_stream, Encoding.UTF8, false, 1024, leaveOpen: true);
            _writer = new StreamWriter(_stream, Encoding.UTF8, 1024, leaveOpen: true) { NewLine = "\n", AutoFlush = true };

            // Receive server handshake FIRST
            string serverHello = await _reader.ReadLineAsync(); // blocks until server sends line
                                                                // Send our handshake
            int rewardsSize = 0, patienceMax = 0, dtMs = Mathf.RoundToInt(dt * 1000f);
            await _writer.WriteLineAsync($"{rewardsSize},{patienceMax},{dtMs}");

            _connected = true;   // <-- only now
            Debug.Log($"[SNNControllerModified] Connected; server said: {serverHello}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"[SNNControllerModified] Failed to connect: {ex.Message}");
            _connected = false;
        }
    }

    private async Task RequestActionAsync()
    {
        if (!_connected || joints == null || joints.Length == 0) return;
        _awaiting = true;
        try
        {
            // --- build feature CSV (3 axes per joint, normalized degrees) ---
            int dof = joints.Length * 3;
            float[] features = new float[dof];
            int idx = 0;
            foreach (var j in joints)
            {
                var e = j.transform.localEulerAngles;
                features[idx++] = NormaliseAngle(e.x);
                features[idx++] = NormaliseAngle(e.y);
                features[idx++] = NormaliseAngle(e.z);
            }
            var sb = new StringBuilder();
            for (int i = 0; i < features.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(features[i].ToString("F6", CultureInfo.InvariantCulture));
            }

            // --- single write then single read (no parallel ops) ---
            await _writer.WriteLineAsync(sb.ToString());            // WRITE
            string line = await _reader.ReadLineAsync();            // READ
            if (!string.IsNullOrEmpty(line))
            {
                var parts = line.Split(',');
                int n = Math.Min(parts.Length, joints.Length * 3);
                float[] acts = new float[n];
                for (int i = 0; i < n; i++)
                    acts[i] = float.Parse(parts[i], CultureInfo.InvariantCulture);

                // apply (3 values per joint)
                int k = 0;
                foreach (var j in joints)
                {
                    if (k + 2 >= n) break;
                    j.transform.localEulerAngles = new Vector3(acts[k++], acts[k++], acts[k++]);
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[SNNControllerModified] Comm error: {ex.Message}");
            _connected = false;
        }
        finally
        {
            _awaiting = false;
        }
    }

    /// <summary>
    /// Apply the predicted orientations to the joints directly.
    /// Expected actions array has three values (degrees) per joint in the same order as inputs.
    /// </summary>
    // --- Use these limits when applying model outputs ---
    private void ApplyAction(float[] actions) {
        Debug.Log("Actions: " + actions);
        if (joints == null || jointLimits == null) return;
        int idx = 0;
        for (int j = 0; j < joints.Length; j++) {
            var lim = jointLimits[Mathf.Clamp(j, 0, jointLimits.Length - 1)];

            float x = Clamp180(actions[idx++], lim.X.min, lim.X.max);
            float y = Clamp180(actions[idx++], lim.Y.min, lim.Y.max);
            float z = Clamp180(actions[idx++], lim.Z.min, lim.Z.max);

            var cur = joints[j].localEulerAngles;

            if (slewDegreesPerSec > 0f) {
                float step = Mathf.Max(0f, slewDegreesPerSec) * Time.deltaTime;
                x = Mathf.MoveTowardsAngle(cur.x, x, step);
                y = Mathf.MoveTowardsAngle(cur.y, y, step);
                z = Mathf.MoveTowardsAngle(cur.z, z, step);
            }

            joints[j].localEulerAngles = new Vector3(x, y, z);
        }
    }

    /// <summary>
    /// Convert any angle from [0,360] range into [-180,180] for network input.
    /// </summary>
    private float NormaliseAngle(float a)
    {
        a = Mathf.Repeat(a + 180f, 360f) - 180f;
        return a;
    }
}