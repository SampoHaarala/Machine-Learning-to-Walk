using System;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;

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
    [Header("Input")]
    // For each action, how many outputs.
    public int outputPerAction = 3;
    public bool useSeperateForPosAndNeg = false;

    // Hysteresis thresholds to prevent rapid toggling
    [Header("Curriculum")]
    [Range(0, 5)] public int lesson = 0;
    [SerializeField] float footRaiseHigh = 0.20f; // meters above ref
    [SerializeField] float footLowerLow = 0.05f; // meters above ref
    [SerializeField] float expectedPushVelocity = 0.5f; // if you use forwardPush

    // Progression tracking
    public float rollingReward = 0f;
    public int stepsInLesson = 0;
    // Keep a note of the previous velocity for counting acceleration.
    private float previousReward = 0;

    // Phase state
    private bool phaseCanSwitch = true;
    private bool leftFootRaising = true;
    public bool forwardPush = false;
    public bool raiseFeet = false;

    // Phase timing
    [SerializeField] float phaseRecentWindow = 0.35f; // seconds
    private float simTime = 0f; // our own physics time
    private float lastPhaseSwitchTime = -999f;
    public int[] lessonStepCounts = { 50, 200, 2000, 1000, 2000, 2000 };
    public float[] lessonRewardGoal = { 0.5f, 0.7f, 0.6f, 0.6f, 0.5f, 0.9f };

    // Store a rolling window of recent episode rewards.
    // A larger window smooths out noise when checking for progress.
    private int rewardsSize = 10;
    private Queue<float> rewards = new Queue<float>();

    // Error
    float[] lastTargetDeg;   // length = DoF

    // Track how many consecutive checks show no meaningful improvement.
    // If this counter exceeds patienceMax, we treat it as a plateau
    // and increase exploration noise to escape it.
    [SerializeField]
    private int patienceMax = 10;
    [SerializeField]
    private int patience = 0;

    // Exploration noise 
    private float exploration = 0f;

    [Header("Network settings")]
    public string host = "127.0.0.1";
    public int port = 6900;

    // List of joints controlled by the neural network.  The size of
    // this array should match the output dimensionality of the Python
    // model (i.e. number of degrees of freedom).  You can assign
    // joint components here via the Inspector.
    [Header("Joint configuration")]
    public CharacterJoint[] joints;
    // axes: 0=X (twist), 1=Y (swing1), 2=Z (swing2)
    private float[,] minDeg;
    private float[,] maxDeg;
    public int degreesOfFreedom = 18;
    // Gains per joint i and axis (x=0,y=1,z=2). Unused = 0.
    // Kp gains per joint (x,y,z). Tune as needed.
    float[,] KpArr = new float[11, 3]
    {
    /* leftLeg       */ { 60f, 60f, 0f },
    /* rightLeg      */ { 60f, 60f, 0f },
    /* leftKnee      */ { 45f,  0f, 0f },
    /* rightKnee     */ { 45f,  0f, 0f },
    /* leftFoot      */ { 35f,  0f, 0f },
    /* rightFoot     */ { 35f,  0f, 0f },
    /* spine2        */ { 50f, 50f, 0f },   // upper spine: moderate stiffness
    /* leftShoulder  */ { 40f, 40f, 0f },   // shoulders softer than hips
    /* rightShoulder */ { 40f, 40f, 0f },
    /* leftArm       */ { 30f,  0f, 0f },   // arms: single-axis by default
    /* rightArm      */ { 30f,  0f, 0f }
    };

    float[,] KdArr = new float[11, 3]
    {
    /* leftLeg       */ { 12f, 12f, 0f },
    /* rightLeg      */ { 12f, 12f, 0f },
    /* leftKnee      */ {  9f,  0f, 0f },
    /* rightKnee     */ {  9f,  0f, 0f },
    /* leftFoot      */ {  7f,  0f, 0f },
    /* rightFoot     */ {  7f,  0f, 0f },
    /* spine2        */ { 10f, 10f, 0f },   // ~0.2*Kp is a good start
    /* leftShoulder  */ {  8f,  8f, 0f },
    /* rightShoulder */ {  8f,  8f, 0f },
    /* leftArm       */ {  6f,  0f, 0f },
    /* rightArm      */ {  6f,  0f, 0f }
    };

    // Optional per-DoF torque clamps (N·m)
    float[,] tauMax = new float[11, 3]
    {
    /* leftLeg       */ { 150f, 150f, 0f },
    /* rightLeg      */ { 150f, 150f, 0f },
    /* leftKnee      */ { 120f,   0f, 0f },
    /* rightKnee     */ { 120f,   0f, 0f },
    /* leftFoot      */ {  80f,   0f, 0f },
    /* rightFoot     */ {  80f,   0f, 0f },
    /* spine2        */ { 120f, 120f, 0f },
    /* leftShoulder  */ { 100f, 100f, 0f },
    /* rightShoulder */ { 100f, 100f, 0f },
    /* leftArm       */ {  80f,   0f, 0f },
    /* rightArm      */ {  80f,   0f, 0f }
    };

    // Rigidbody for calculating velocity.
    [Header("Rigidbodies")]
    public Rigidbody rb;
    public Rigidbody leftFoot;
    public Rigidbody rightFoot;
    // The spine is needed for making sure that the model straightens its upright.
    public GameObject spine;

    [Header("Feet Colliders")]
    public Collider leftFootCol;
    public Collider rightFootCol;

    // Internal TCP client and stream
    private TcpClient _client;
    private NetworkStream _stream;
    private StreamReader _reader;
    private StreamWriter _writer;
    private byte[] _readBuffer = new byte[4096];
    private StringBuilder _recvBuilder = new StringBuilder();

    // Simulation timing
    private bool _connected = false;
    [SerializeField, Header("Variables")]
    private float dt = 0.02f;
    // Actions
    private float[] pendingAction = null;
    private volatile bool awaitingAction = false;
    public float actionMultiplier = 4;
    public bool useTorque = false;
    public bool useGravity = false;
    [SerializeField] bool actionsAreDegrees = false;        // set true if Python sends absolute deg
    [SerializeField] float rotationSlewDegPerSec = 360f;    // 0 = snap; otherwise max deg/s change

    void Awake()
    {
        // Stop automatic simulation. This way simulation speed can be customised.
        Physics.simulationMode = SimulationMode.Script; // (was commented out)

        ResetPhaseState();

        if (!spine) Debug.Log("Spine gameobject is missing!");

        minDeg = new float[joints.Length, 3];
        maxDeg = new float[joints.Length, 3];
        lastTargetDeg = new float[degreesOfFreedom + 2];

        Rigidbody[] rigidbodies = GetComponentsInChildren<Rigidbody>();
        foreach (Rigidbody rigidbody in rigidbodies) rigidbody.useGravity = useGravity;

        // Set physics limits.
        foreach (Rigidbody jrb in gameObject.GetComponentsInChildren<Rigidbody>())
        {
            Debug.Log(jrb.name);
            jrb.maxAngularVelocity = 10f;
            jrb.maxDepenetrationVelocity = 5f;
        }

        // For debugging.
        if (joints == null)
            joints = gameObject.GetComponentsInChildren<CharacterJoint>();

        int dofs = 0;
        foreach (CharacterJoint joint in joints)
        {
            // Make sure how many joints exists and how many DoF they have.
            Debug.Log("Found joint: " + joint.name);
            for (int i = 0; i < 3; i++)
            {
                if (!IsAxisLocked(joint, i))
                {
                    Debug.Log("Axis " + i + " of joint " + joint + " is moving.");
                    dofs++;
                }
                else Debug.Log("Axis " + i + " of joint " + joint + " is locked.");
            }
        }
        Debug.Log("The model has " + dofs + " degrees of freedom.");
        _ = ConnectAndHandshakeAsync();
        ConfigureLesson(lesson);
    }

    void OnDestroy()
    {
        // Clean up the connection when the object is destroyed
        if (_stream != null)
            _stream.Close();
        if (_client != null)
            _client.Close();
    }

    void Update()
    {
        if (!_connected) return;

        // 

        // If an action just arrived, apply and step exactly once.
        if (pendingAction != null)
        {
            if (patience < patienceMax)
            {
                exploration = 0;
            }
            else
            {
                ExplorationNoise(0.8f);
            }
            ApplyAction(pendingAction);
            pendingAction = null;
            // Update simulation once communication is done.
            Physics.Simulate(dt);
            simTime += dt; // advance our clock
            stepsInLesson++;
        }

        // Immediately request the next action for the next tick
        RequestActionAsync();

        if (forwardPush)
        {
            rb.velocity = new Vector3(0, 0, 1f);
        }

        if (raiseFeet)
        {
            if (leftFootRaising) leftFoot.velocity = new Vector3(0, 0.5f, 0);
            else rightFoot.velocity = new Vector3(0, 0.5f, 0);
        }
        // Else: still waiting for SNN — do nothing this frame (no accumulation).
    }

    // ------------------- Networking helpers -------------------


    /// <summary>
    /// Establishes TCP connection and performs handshake with the SNN server.
    /// Reads rewards_size, patience_max, and dt from the server.
    /// </summary>
    async Task ConnectAndHandshakeAsync()
    {
        try
        {
            // Force IPv4 and connect to explicit loopback
            _client = new TcpClient(AddressFamily.InterNetwork);
            await _client.ConnectAsync(IPAddress.Loopback, port);

            _stream = _client.GetStream();
            _reader = new StreamReader(_stream);
            _writer = new StreamWriter(_stream) { NewLine = "\n", AutoFlush = true };
            _connected = true;
            Debug.Log($"[SNNController] Connected to SNN server at 127.0.0.1:{port}");

            // ---- Handshake receive ----
            string line = await _reader.ReadLineAsync();
            if (string.IsNullOrEmpty(line))
            {
                Debug.LogError("[SNNController] Handshake failed: no data");
                _connected = false;
                return;
            }

            try
            {
                // Python sends: rewards_size,patience_max,dt
                string[] parts = line.Split(',');
                if (parts.Length >= 3)
                {
                    rewardsSize = int.Parse(parts[0], CultureInfo.InvariantCulture);
                    patienceMax = int.Parse(parts[1], CultureInfo.InvariantCulture);
                    dt = float.Parse(parts[2], CultureInfo.InvariantCulture) / 100; // seconds
                    Debug.Log("[SNNController] Handshake OK. Sending first state.");
                }
                else
                {
                    Debug.LogError("[SNNController] Handshake malformed: " + line);
                    _connected = false;
                    return;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError("[SNNController] Handshake parse error: " + ex.Message);
                _connected = false;
                return;
            }

            // Kick off the first request
            RequestActionAsync();
        }
        catch (Exception ex)
        {
            Debug.LogError($"[SNNController] Could not connect to SNN server: {ex.Message}");
            _connected = false;
        }
    }

    /// <summary>
    /// Sends the current simulation state (features + reward) to the
    /// SNN server and starts an async read for the next action.
    /// </summary>
    void RequestActionAsync()
    {
        if (awaitingAction || !_connected) return;
        awaitingAction = true;

        // 1) Build observation line from current simulation state
        // Get all features from the simulation.
        float[] features = null;
        if (useSeperateForPosAndNeg)
        {
            features = new float[degreesOfFreedom * 2];
            if (joints != null && joints.Length > 0)
            {
                int j = 0;
                for (int i = 0; i < joints.Length && j < features.Length; i++)
                {
                    // Get all angular velocity data of the joint.
                    Rigidbody jrb = joints[i].GetComponent<Rigidbody>();
                    Vector3 wLocal = joints[i].transform.InverseTransformDirection(jrb.angularVelocity);

                    // normalize by a realistic cap (e.g., 10 rad/s) and clamp to [-1,1]
                    float a = 0f;
                    float b = 0f;
                    if (!IsAxisLocked(joints[i], 0) && j < features.Length)
                    {
                        // Seperate positive and negative feedback to two outputs
                        a = wLocal.x;
                        b = Mathf.Abs(a);
                        if (a == b)
                        {
                            features[j++] = a; features[j++] = 0;
                        }
                        else
                        {
                            features[j++] = 0; features[j++] = b;
                        }
                    }
                    if (!IsAxisLocked(joints[i], 1) && j < features.Length)
                    {
                        // Seperate positive and negative feedback to two outputs
                        a = wLocal.y;
                        b = Mathf.Abs(a);
                        if (a == b)
                        {
                            features[j++] = a; features[j++] = 0;
                        }
                        else
                        {
                            features[j++] = 0; features[j++] = b;
                        }
                    }
                    if (!IsAxisLocked(joints[i], 2) && j < features.Length)
                    {
                        // Seperate positive and negative feedback to two outputs
                        a = wLocal.z;
                        b = Mathf.Abs(a);
                        if (a == b)
                        {
                            features[j++] = a; features[j++] = 0;
                        }
                        else
                        {
                            features[j++] = 0; features[j++] = b;
                        }
                    }
                }
            }
        }
        else
        {
            features = new float[degreesOfFreedom];
            if (joints != null && joints.Length > 0)
            {
                int j = 0;
                for (int i = 0; i < joints.Length && j < features.Length; i++)
                {
                    Transform jtr = joints[i].transform;
                    // Get rotational data of the joint.
                    if (!IsAxisLocked(joints[i], 0) && j < features.Length)
                        features[j++] = jtr.localEulerAngles.x;
                    if (!IsAxisLocked(joints[i], 1) && j < features.Length)
                        features[j++] = jtr.localEulerAngles.y;
                    if (!IsAxisLocked(joints[i], 2) && j < features.Length)
                        features[j++] = jtr.localEulerAngles.z;
                }
            }
        }

        // REWARD
        float reward = 0f;
        if (rewards.Count < rewardsSize)
        {
            // Reward depends on the lesson.
            reward = CurriculumReward(lesson);
            rollingReward = 0.99f * rollingReward + 0.01f * reward;
            rewards.Enqueue(reward);
        }
        else
        {
            if (IsPlateu()) patience++;
            else patience = 0;

            reward = CurriculumReward(lesson);
            rollingReward = 0.99f * rollingReward + 0.01f * reward;
            rewards.Enqueue(reward);
            rewards.Dequeue();
        }

        if (stepsInLesson > lessonStepCounts[lesson] && rollingReward > lessonRewardGoal[lesson] && lesson < 5)
        {
            if (lesson < 6) lesson++;
            else Debug.Log("Model working.");
            ConfigureLesson(lesson);
        }

        // Error is received from acceleration.
        float error = ComputeTrackingError();
        // Keep the previous reward for the next tick.
        previousReward = reward;

        // Encode features and reward into a comma-separated string
        // terminated by a newline. The Python server will parse this
        // message into floats.
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < features.Length; i++)
        {
            sb.Append(features[i].ToString("F6", CultureInfo.InvariantCulture));
            sb.Append(',');
        }
        // Foot contacts
        sb.Append((IsFootTouching(leftFootCol) ? 1 : 0).ToString("F6", CultureInfo.InvariantCulture));
        sb.Append(',');
        sb.Append((IsFootTouching(rightFootCol) ? 1 : 0).ToString("F6", CultureInfo.InvariantCulture));
        sb.Append(',');

        // Reward and error
        sb.Append(reward.ToString("F6", CultureInfo.InvariantCulture));
        sb.Append(',');
        sb.Append(error.ToString("F6", CultureInfo.InvariantCulture));
        // writer has NewLine = "\n"
        _ = _writer.WriteLineAsync(sb.ToString());

        // 2) Read response off the main thread; enqueue for Update()
        _ = Task.Run(async () =>
        {
            try
            {
                string line = await _reader.ReadLineAsync();
                float[] act = null;
                if (!string.IsNullOrEmpty(line))
                {
                    string[] parts = line.Split(',');
                    int actionCount = degreesOfFreedom;
                    act = new float[actionCount];
                    for (int i = 0; i < actionCount; i++)
                        for (int j = 0; j < outputPerAction; j++)
                            act[i] += float.Parse(parts[i + j], NumberStyles.Float, CultureInfo.InvariantCulture);
                }
                // Marshal back: store for Update()
                Debug.Log("Actions: " + line);
                pendingAction = act;
            }
            catch (Exception ex)
            {
                Debug.LogError("[SNNController] Receive error: " + ex.Message);
                pendingAction = null;
                _connected = false;
            }
            finally
            {
                awaitingAction = false;
            }
        });
    }

    /// <summary>
    /// Apply a scalar action value to a joint.  This method
    /// sets joint motor targets based on the network's output.  
    /// </summary>
    /// <param name="joint">The joint component to control.</param>
    /// <param name="action">The network's processed output for this joint.</param>
    private void ApplyTorqueToJoint(CharacterJoint joint, Vector3 torque)
    {
        // TO-DO Implement
        Rigidbody jrb = joint.GetComponent<Rigidbody>();
        jrb.AddTorque(actionMultiplier * torque, ForceMode.Acceleration);
    }

    private void ApplyAction(float[] actions)
    {
        if (actions == null || joints == null) return;

        // Parse actions into per-axis forces using your existing IsAxisLocked logic.
        if (useTorque)
        {
            if (useSeperateForPosAndNeg)
            {
                int j = 0;
                int actionCount = actions.Length;
                for (int i = 0; i < joints.Length; i++)
                {
                    Vector3 torque = new Vector3();
                    float a = 0f;
                    float b = 0f;
                    if (!IsAxisLocked(joints[i], 0) && i + j < actionCount)
                    {
                        a = actions[i + j]; j++; // Positive
                        b = actions[i + j]; j++; // Negative
                        torque.x = actionMultiplier * Mathf.Clamp(a - b, -1f, 1f);
                    }
                    if (!IsAxisLocked(joints[i], 1) && i + j < actionCount)
                    {
                        a = actions[i + j]; j++;
                        b = actions[i + j]; j++;
                        torque.y = actionMultiplier * Mathf.Clamp(a - b, -1f, 1f);
                    }
                    if (!IsAxisLocked(joints[i], 2) && i + j < actionCount)
                    {
                        a = actions[i + j]; j++;
                        b = actions[i + j]; j++;
                        torque.z = actionMultiplier * Mathf.Clamp(a - b, -1f, 1f);
                    }
                    torque += new Vector3(exploration, exploration, exploration);
                    ApplyTorqueToJoint(joints[i], torque);
                }
            }
            else
            {
                int j = 0;
                for (int i = 0; i < joints.Length; i++)
                {
                    CharacterJoint ji = joints[i];

                    for (int axis = 0; axis < 3; axis++)
                    {
                        Vector3 torque = new Vector3();
                        if (!IsAxisLocked(ji, axis) && j < actions.Length)
                        {
                            float a = Mathf.Clamp(actions[j], -1f, 1f); j++;

                            // Map action -> target angle (deg) within joint limits:
                            float targetDeg = Mathf.Lerp(minDeg[i, axis], maxDeg[i, axis], (a + 1f) * 0.5f);
                            lastTargetDeg[j - 1] = targetDeg;

                            // PD control
                            float thetaDeg = GetCurrentAngleDeg(ji.transform, axis);
                            float eRad = Mathf.DeltaAngle(thetaDeg, targetDeg) * Mathf.Deg2Rad; // shortest signed error
                            float w = GetAngularVelRad(ji.GetComponent<Rigidbody>(), axis);     // rad/s about axis

                            float Kp = KpArr[i, axis];      // tune per DoF
                            float Kd = KdArr[i, axis];      // tune per DoF
                            float tau = Kp * eRad - Kd * w; // N·m

                            // clamp torque
                            tau = Mathf.Clamp(tau, -tauMax[i, axis], tauMax[i, axis]);

                            // axis unit vectors in joint local or world space (consistent with your angle/velocity)
                            Vector3 axisVec = GetAxisVector(ji.transform, axis); // e.g., local right/up/forward

                            // accumulate torque vector
                            torque += axisVec.normalized * tau;
                        }
                        ApplyTorqueToJoint(ji, torque);
                    }
                }
            }
        }
        else
        {
            int j = 0;                             // index into actions & lastTargetDeg
            float maxStep = rotationSlewDegPerSec * Time.fixedDeltaTime;

            for (int i = 0; i < joints.Length; i++)
            {
                var ji = joints[i];
                Transform tr = ji.transform;

                for (int axis = 0; axis < 3; axis++)
                {
                    if (IsAxisLocked(ji, axis)) continue;   // only unlocked (controlled) axes
                    if (j >= actions.Length) break;

                    float a = actions[j++];

                    // Compute target angle in degrees
                    float min = minDeg[i, axis];
                    float max = maxDeg[i, axis];
                    float targetDeg;

                    if (actionsAreDegrees)
                    {
                        targetDeg = Mathf.Clamp(a, min, max);   // already degrees
                    }
                    else
                    {
                        // a in [-1,1] → map to [min,max]
                        float u = Mathf.Clamp(a, -1f, 1f);
                        targetDeg = Mathf.Lerp(min, max, (u + 1f) * 0.5f);
                    }

                    // Store for tracking error in the same order we consumed actions
                    lastTargetDeg[j - 1] = targetDeg;

                    // Apply: optionally slew to avoid teleport; else snap
                    float currDeg = GetCurrentAngleDeg(tr, axis);
                    float nextDeg = rotationSlewDegPerSec > 0f ? Mathf.MoveTowardsAngle(currDeg, targetDeg, maxStep) : targetDeg;

                    SetLocalAngleDeg(tr, axis, nextDeg);
                }
            }
        }
    }

    /// <summary>
    /// Check if axis is locked. 
    /// </summary>
    /// <param name="j">The joint component to check.</param>
    /// <param name="axis">The axis to be checked.</param>
    bool IsAxisLocked(CharacterJoint j, int axis)
    {
        switch (axis)
        {
            case 0: return Mathf.Approximately(j.lowTwistLimit.limit, 0f) && Mathf.Approximately(j.highTwistLimit.limit, 0f);
            case 1: return Mathf.Approximately(j.swing1Limit.limit, 0f);
            case 2: return Mathf.Approximately(j.swing2Limit.limit, 0f);
            default: return true;
        }
    }

    bool IsPlateu()
    {
        float[] rs = rewards.ToArray();
        float sumY = 0f; float sumXY = 0f;
        for (int i = 0; i < rewardsSize; i++)
        {
            float y = rs[i];
            sumY += y; // Sum of rewards
            sumXY += i * y;
        }

        float xBar = (rewardsSize - 1) * 0.5f;
        float yBar = sumY / rewardsSize;                                  // Mean of rewards
        float Sxx = rewardsSize * (rewardsSize * rewardsSize - 1) / 12f;  // Covariance of rewards
        float Sxy = sumXY - rewardsSize * xBar * yBar;

        float slope = Sxx > 0f ? Sxy / Sxx : 0f;
        if (Mathf.Abs(slope) < 0.007) return true;
        return false;
    }

    /// <summary>
    /// Ornstein–Uhlenbeck noise for smooth exploration in continuous actions.
    /// </summary>
    /// <param name="dt">Delta time (usually Time.fixedDeltaTime)</param>
    /// <param name="theta">Mean reversion rate (0.15 is common)</param>
    /// <param name="sigma">Noise volatility (0.2 typical; increase on plateau)</param>
    /// <param name="mu">Long-term mean (normally 0)</param>
    /// <returns>Noise value to add to an action</returns>
    float ExplorationNoise(float dt, float theta = 0.15f, float sigma = 0.2f, float mu = 0f)
    {
        // Box–Muller transform for standard normal
        float u1 = 1f - UnityEngine.Random.value;
        float u2 = 1f - UnityEngine.Random.value;
        float randStdNormal = Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Sin(2f * Mathf.PI * u2);

        // Ornstein–Uhlenbeck process update
        exploration += theta * (mu - exploration) * dt + sigma * Mathf.Sqrt(dt) * randStdNormal;
        return exploration;
    }

    void ConfigureLesson(int lesson)
    {
        // Clear first
        rb.constraints = RigidbodyConstraints.None;
        forwardPush = false;
        stepsInLesson = 0;
        rollingReward = 0f;
        leftFootRaising = true;
        phaseCanSwitch = true;
        exploration = 0;

        switch (lesson)
        {
            case 0: // upright only
                rb.constraints = RigidbodyConstraints.FreezePositionY
                               | RigidbodyConstraints.FreezePositionX
                               | RigidbodyConstraints.FreezePositionZ;
                break;

            case 1: // Upright, pelvis free in Y only
                rb.constraints = RigidbodyConstraints.FreezePositionX
                               | RigidbodyConstraints.FreezePositionZ;
                break;

            case 2: // raise legs, pelvis fixed in space
                rb.constraints = RigidbodyConstraints.FreezePositionX
                               | RigidbodyConstraints.FreezePositionY
                               | RigidbodyConstraints.FreezePositionZ;
                // Help the feet to raise.
                raiseFeet = true;
                break;

            case 3: // raise legs, pelvis free in Y only
                rb.constraints = RigidbodyConstraints.FreezePositionX
                               | RigidbodyConstraints.FreezePositionZ;
                break;

            case 4: // start moving forward with help
                rb.constraints = RigidbodyConstraints.FreezePositionX
                               | RigidbodyConstraints.FreezePositionY;
                forwardPush = true; // if you must use it
                break;
            case 5: // Free
                rb.constraints = RigidbodyConstraints.FreezePositionX
                               | RigidbodyConstraints.FreezePositionY
                               | RigidbodyConstraints.FreezePositionZ;
                break;
        }

        // Optional: reset phase state/hysteresis
        leftFootRaising = true;
        phaseCanSwitch = true;
    }

    float FootHeight(Transform foot, Transform pelvis) =>
    Vector3.Dot(foot.position - pelvis.position, Vector3.up); // relative to pelvis

    float CurriculumReward(int lesson)
    {
        float reward = 0f;

        // Common locals
        float dt = Time.fixedDeltaTime;
        float leftH = FootHeight(leftFoot.transform, rb.transform);
        float rightH = FootHeight(rightFoot.transform, rb.transform);
        float hipsH = rb.transform.localPosition.y;

        switch (lesson)
        {
            case 0: // upright only
                // Upright reward in [0,1]
                // Use pelvis/spine “up” vs world up
                reward = Vector3.Dot(-spine.transform.right, Vector3.up);
                // Small penalty for large angular velocity (don’t learn to vibrate)
                reward -= 0.01f * rb.angularVelocity.sqrMagnitude;
                return reward;

            case 1: // Upright, pelvis free in Y only
                // Upright reward in [0,1]
                // Use pelvis/spine “up” vs world up
                reward = 0.7f * Vector3.Dot(-spine.transform.right, Vector3.up) + 0.3f * hipsH / 0.7f;
                // Small penalty for large angular velocity (don’t learn to vibrate)
                reward -= 0.01f * rb.angularVelocity.sqrMagnitude;
                return reward;

            case 2: // raise legs, pelvis fixed in space
                // Velocity shaping: up foot should move up, other down.
                // NOTE: reward velocity + height target to avoid jitter exploits.
                Vector3 vL = leftFoot.velocity;
                Vector3 vR = rightFoot.velocity;
                if (leftFootRaising)
                {
                    float velTerm = Vector3.Dot(vL, Vector3.up) - Vector3.Dot(vR, Vector3.up); // up minus up = up/down contrast
                    float heightTerm = Mathf.Clamp01(leftH / footRaiseHigh);                  // approach target height
                    reward = 0.5f * velTerm + 0.5f * heightTerm + 0.25f * Vector3.Dot(-spine.transform.right, Vector3.up);
                    // Switch with hysteresis: only switch once above high, then require other foot to go below low
                    if (phaseCanSwitch && leftH >= footRaiseHigh) { FlipToRight(); }
                    if (!leftFootRaising && rightH <= footLowerLow) phaseCanSwitch = true;
                }
                else
                {
                    float velTerm = Vector3.Dot(vR, Vector3.up) - Vector3.Dot(vL, Vector3.up);
                    float heightTerm = Mathf.Clamp01(rightH / footRaiseHigh);
                    reward = 0.5f * velTerm + 0.5f * heightTerm + 0.25f * Vector3.Dot(-spine.transform.right, Vector3.up);
                    if (phaseCanSwitch && rightH >= footRaiseHigh) { FlipToLeft(); }
                    if (leftFootRaising && leftH <= footLowerLow) phaseCanSwitch = true;
                }
                // Light contact bonus to discourage foot-drag when not the “up” foot
                reward += (leftFootRaising ? !IsFootTouching(leftFootCol) : !IsFootTouching(rightFootCol)) ? 0.05f : 0f;
                return reward;

            case 3: // raise legs, pelvis free in Y only
                // Upright reward in [0,1]
                // Use pelvis/spine “up” vs world up
                reward = 0.7f * Vector3.Dot(-spine.transform.right, Vector3.up) + 0.3f * hipsH / 0.7f;
                // Small penalty for large angular velocity (don’t learn to vibrate)
                reward -= 0.01f * rb.angularVelocity.sqrMagnitude;
                return reward;

            case 4: // start moving forward with help
                {
                    // Same as 1, but pelvis can move in Y. Reuse shaping.
                    // (Consider small penalty for COM lateral drift)
                    float baseR = CurriculumReward(1); // or factor out shared code
                    reward = baseR - 0.02f * new Vector2(rb.velocity.x, rb.velocity.z).sqrMagnitude + 0.25f * Vector3.Dot(-spine.transform.right, Vector3.up); // keep from skating
                    return reward;
                }

            case 5: // Free
                {
                    // Forward velocity reward – subtract the push baseline to avoid “free points”
                    float vFwd = Vector3.Dot(rb.velocity, transform.forward);
                    reward = Mathf.Max(0f, vFwd);

                    // Penalize tumbling
                    reward += Mathf.Clamp01(Vector3.Dot(spine.transform.up, Vector3.up)) - 1f + 0.25f * Vector3.Dot(spine.transform.up, Vector3.up); // adds 0..-1
                    return reward;
                }

            default:
                return 0f;
        }
    }

    bool IsFootTouching(Collider foot)
    {
        return Physics.CheckBox(foot.bounds.center, foot.bounds.extents, foot.transform.rotation);
    }

    private bool PhaseRecentlySwitched()
    {
        return (simTime - lastPhaseSwitchTime) <= phaseRecentWindow;
    }

    private void FlipToRight()
    {
        leftFootRaising = false;
        phaseCanSwitch = false;
        lastPhaseSwitchTime = simTime; // stamp with our clock
    }

    private void FlipToLeft()
    {
        leftFootRaising = true;
        phaseCanSwitch = false;
        lastPhaseSwitchTime = simTime; // stamp with our clock
    }

    // reset on episode / lesson change
    private void ResetPhaseState()
    {
        leftFootRaising = true;
        phaseCanSwitch = true;
        lastPhaseSwitchTime = -999f;
        simTime = 0f; // or keep running across episodes—your choice
    }

    void OnDrawGizmos()
    {
        if (!rb) return;
        Gizmos.color = Color.cyan;
        Gizmos.DrawLine(rb.transform.position, rb.transform.position - rb.transform.right * 0.5f);
    }

    void BakeCharacterJointLimits(float marginDeg = 2f)
    {
        for (int i = 0; i < joints.Length; i++)
        {
            var cj = joints[i].GetComponent<CharacterJoint>();

            // X (twist)
            minDeg[i, 0] = cj.lowTwistLimit.limit + marginDeg;
            maxDeg[i, 0] = cj.highTwistLimit.limit - marginDeg;

            // Y (swing1) is symmetric
            float s1 = cj.swing1Limit.limit;
            minDeg[i, 1] = -s1 + marginDeg;
            maxDeg[i, 1] = s1 - marginDeg;

            // Z (swing2) is symmetric
            float s2 = cj.swing2Limit.limit;
            minDeg[i, 2] = -s2 + marginDeg;
            maxDeg[i, 2] = s2 - marginDeg;
        }
    }
    Vector3 GetAxisVector(Transform joint, int axis)
    {
        // axis: 0 = twist(X), 1 = swing1(Y), 2 = swing2(Z)
        var cj = joint.GetComponent<CharacterJoint>();
        if (cj != null)
        {
            // Local basis from CharacterJoint’s axis definitions
            // X = twist axis, Y = swing1 axis, Z = orthogonal (swing2)
            Vector3 x = cj.axis.sqrMagnitude > 0f ? cj.axis.normalized : Vector3.right;
            Vector3 y = cj.swingAxis.sqrMagnitude > 0f ? cj.swingAxis.normalized : Vector3.up;
            Vector3 z = Vector3.Cross(x, y).normalized;   // make Z orthogonal
            y = Vector3.Cross(z, x).normalized;           // re-orthonormalize Y

            switch (axis)
            {
                case 0: return joint.TransformDirection(x); // world-space X
                case 1: return joint.TransformDirection(y); // world-space Y
                case 2: return joint.TransformDirection(z); // world-space Z
                default: return Vector3.zero;
            }
        }

        // Fallback: assume local XYZ
        switch (Mathf.Clamp(axis, 0, 2))
        {
            case 0: return joint.TransformDirection(Vector3.right);
            case 1: return joint.TransformDirection(Vector3.up);
            default: return joint.TransformDirection(Vector3.forward);
        }
    }

    // Signed local angle (deg) around one axis of a joint Transform.
    // axis: 0=x, 1=y, 2=z
    float GetCurrentAngleDeg(Transform joint, int axis)
    {
        // Local Euler angles are [0..360). DeltaAngle gives signed [-180..180]
        Vector3 e = joint.localEulerAngles;
        float raw = (axis == 0) ? e.x : (axis == 1) ? e.y : e.z;
        return Mathf.DeltaAngle(0f, raw);
    }

    // axis: 0=x, 1=y, 2=z  (joint local axes)
    // returns signed angular speed (rad/s) about that axis
    float GetAngularVelRad(Rigidbody rb, int axis)
    {
        if (rb == null) return 0f;
        // local axis in world space
        Vector3 axisWorld = (axis == 0) ? rb.transform.right
                            : (axis == 1) ? rb.transform.up
                                          : rb.transform.forward;
        // world angular vel projected to that axis (rad/s, signed)
        return Vector3.Dot(rb.angularVelocity, axisWorld);
    }

    float ComputeTrackingError()
    {
        float sum = 0f;
        int N = 0;
        int j = 0;
        for (int i = 0; i < joints.Length; i++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                if (IsAxisLocked(joints[i], axis)) continue; // we control only unlocked axes
                float theta = GetCurrentAngleDeg(joints[i].transform, axis);     // signed current deg
                float target = lastTargetDeg[j++];                     // stored target deg
                // shortest signed error (deg)
                float eDeg = Mathf.DeltaAngle(theta, target);
                // normalize by joint range so all DoFs are comparable
                float range = Mathf.Max(5f, maxDeg[i, axis] - minDeg[i, axis]); // avoid /0
                float eNorm = eDeg / range;
                sum += eNorm * eNorm;
                N++;
            }
        }
        return (N > 0) ? (sum / N) : 0f;   // MSE over DoFs (unitless)
    }

    // Helper: set a single local Euler component (deg)
    static void SetLocalAngleDeg(Transform t, int axis, float targetDeg)
    {
        Vector3 e = t.localEulerAngles;
        if (axis == 0) e.x = targetDeg;
        else if (axis == 1) e.y = targetDeg;
        else e.z = targetDeg;
        t.localEulerAngles = e;
    }
}
