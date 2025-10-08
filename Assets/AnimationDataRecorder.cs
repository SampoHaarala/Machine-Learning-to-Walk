using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Records the rotations (and optionally rotations) of specific bones on a humanoid during a looping animation.
/// Attach this script to any GameObject in your scene and populate the bone fields via the Inspector.
/// When the recording duration elapses the collected data is saved to a JSON file in the persistent data path.
/// You can adjust the recordDuration to match the length of your animation loop (e.g. ~0.77 seconds for a 23‑frame loop at 30 FPS).
/// </summary>
// ... existing usings ...

public class AnimationDataRecorder : MonoBehaviour
{
    [Header("Transforms to record (lower body)")]
    public Transform hip;
    public Transform leftLeg;
    public Transform rightLeg;
    public Transform leftKnee;
    public Transform rightKnee;
    public Transform leftAnkle;
    public Transform rightAnkle;

    [Header("Transforms to record (upper body)")]
    public Transform spine2;
    public Transform leftShoulder;
    public Transform rightShoulder;
    public Transform leftArm;   // upper arm or forearm – your rig
    public Transform rightArm;

    [Header("Axis selection (enable per bone)")]
    // Hip/legs (old behaviour by default)
    public bool hipX = true;
    public bool hipY = true, hipZ = false;
    public bool leftLegX = true, leftLegY = true, leftLegZ = false;
    public bool rightLegX = true, rightLegY = true, rightLegZ = false;
    public bool leftKneeX = true, leftKneeY = false, leftKneeZ = false;
    public bool rightKneeX = true, rightKneeY = false, rightKneeZ = false;
    public bool leftAnkleX = true, leftAnkleY = false, leftAnkleZ = false;
    public bool rightAnkleX = true, rightAnkleY = false, rightAnkleZ = false;

    // New upper body defaults (tweak as needed)
    public bool spine2X = true, spine2Y = true, spine2Z = false;
    public bool leftShoulderX = true, leftShoulderY = true, leftShoulderZ = false;
    public bool rightShoulderX = true, rightShoulderY = true, rightShoulderZ = false;
    public bool leftArmX = true, leftArmY = false, leftArmZ = false;
    public bool rightArmX = true, rightArmY = false, rightArmZ = false;

    [Header("Recording settings")]
    public float recordDuration = 1f;
    public float sampleInterval = 0f;

    [Header("Classification")]
    [Tooltip("Label for this animation (e.g., 'standing', 'walking', 'running').")]
    public string label = "walking";

    private readonly List<FrameData> _frames = new();
    private float _elapsed;
    private float _nextSampleTime;
    private bool _isRecording;

    public Collider rightFootCol;
    public Collider leftFootCol;
    private bool rightFootTouching = true;
    private bool leftFootTouching = true;
    private Animator animator;
    private string firstClipName;
    private int currentAnimation = 0;

    private void Start()
    {
        _isRecording = true;
        _elapsed = 0f;
        _nextSampleTime = 0f;
        _frames.Clear();

        animator = GetComponent<Animator>();
        firstClipName = animator.GetCurrentAnimatorClipInfo(0)[0].clip.name;
    }

    private void Update()
    {
        if (!_isRecording) return;

        animator.SetBool("next", false);
        _elapsed += Time.deltaTime;

        if (sampleInterval <= 0f || _elapsed >= _nextSampleTime)
        {
            RecordFrame();
            if (sampleInterval > 0f) _nextSampleTime += sampleInterval;
        }

        if (_elapsed >= recordDuration)
        {
            if (animator.GetCurrentAnimatorClipInfo(0)[0].clip.name == firstClipName && currentAnimation != 0) _isRecording = false;
            animator.SetBool("next", true);
            SaveToJson();
            _frames.Clear();
            _elapsed = 0f;
            _nextSampleTime = 0f;
            currentAnimation++;
        }
    }

    // --- Helper: append selected axes from a Transform's localEulerAngles ---
    private static void AppendAxes(Transform t, bool ax, bool ay, bool az, List<float> dst)
    {
        if (!t) return;
        var e = t.localEulerAngles; // [0..360)
        if (ax) dst.Add(NormaliseAngle(e.x));
        if (ay) dst.Add(NormaliseAngle(e.y));
        if (az) dst.Add(NormaliseAngle(e.z));
    }

    private void RecordFrame()
    {
        var rot = new List<float>(32); // reserve some space

        // --- ORDER (documented for downstream parsing) ---
        // hip[x? y? z?],
        // leftLeg[x? y? z?], rightLeg[x? y? z?],
        // leftKnee[x? y? z?], rightKnee[x? y? z?],
        // leftAnkle[x? y? z?], rightAnkle[x? y? z?],
        // spine2[x? y? z?],
        // leftShoulder[x? y? z?], rightShoulder[x? y? z?],
        // leftArm[x? y? z?], rightArm[x? y? z?],
        // foot contacts (L,R) appended last as 0/1.

        AppendAxes(hip, hipX, hipY, hipZ, rot);
        AppendAxes(leftLeg, leftLegX, leftLegY, leftLegZ, rot);
        AppendAxes(rightLeg, rightLegX, rightLegY, rightLegZ, rot);
        AppendAxes(leftKnee, leftKneeX, leftKneeY, leftKneeZ, rot);
        AppendAxes(rightKnee, rightKneeX, rightKneeY, rightKneeZ, rot);
        AppendAxes(leftAnkle, leftAnkleX, leftAnkleY, leftAnkleZ, rot);
        AppendAxes(rightAnkle, rightAnkleX, rightAnkleY, rightAnkleZ, rot);

        // NEW upper body
        AppendAxes(spine2, spine2X, spine2Y, spine2Z, rot);
        AppendAxes(leftShoulder, leftShoulderX, leftShoulderY, leftShoulderZ, rot);
        AppendAxes(rightShoulder, rightShoulderX, rightShoulderY, rightShoulderZ, rot);
        AppendAxes(leftArm, leftArmX, leftArmY, leftArmZ, rot);
        AppendAxes(rightArm, rightArmX, rightArmY, rightArmZ, rot);

        // Contacts at the end (keep this order for downstream)
        // rot.Add(leftFootTouching ? 1f : 0f);
        // rot.Add(rightFootTouching ? 1f : 0f);

        var frame = new FrameData
        {
            time = _elapsed,
            rotations = rot.ToArray()
        };
        _frames.Add(frame);
    }

    /// <summary>
    /// Serializes the recorded frames to JSON and writes them to a file on disk.
    /// The output file will be created in Application.persistentDataPath.
    /// A label is included to support supervised classification.
    /// </summary>
    private void SaveToJson()
    {
        // Wrap the list in a container so Unity's JsonUtility can serialize it and include the label
        var container = new FrameDataContainer { label = label, frames = _frames };
        string json = JsonUtility.ToJson(container, prettyPrint: true);
        // Build a filename based on date/time for uniqueness
        string fileName = $"animation_data_{System.DateTime.Now:yyyyMMdd_HHmmss}_{currentAnimation}.json";
        // Save to persistent data path so that it works across platforms
        string filePath = Path.Combine("C:/Users/sampo/OneDrive/Documents/Machine Learning to Walk/Assets", fileName);
        try
        {
            File.WriteAllText(filePath, json);
            Debug.Log($"Animation data saved to {filePath}");
        }
        catch (IOException e)
        {
            Debug.LogError($"Failed to write animation data: {e.Message}");
        }
    }

    /// <summary>
    /// Represents a single frame of animation data.
    /// </summary>
    [System.Serializable]
    private class FrameData
    {
        public float time;
        public float[] rotations;
    }

        /// <summary>
    /// Wrapper class so that JsonUtility can serialize a list of frames and a label.
    /// </summary>
    [System.Serializable]
    private class FrameDataContainer
    {
        public string label;
        public List<FrameData> frames;
    }

    void OnCollisionStay(Collision c)
    {
        if (c.GetContact(0).thisCollider == leftFootCol) leftFootTouching = true;
        else if (c.GetContact(0).thisCollider == rightFootCol) rightFootTouching = true;
    }
    void OnCollisionExit(Collision c) { leftFootTouching = false; rightFootTouching = false; }

    static private float NormaliseAngle(float a)
    {
        float n = Mathf.Repeat(a + 180f, 360f) - 180f;
        return n;
    }
}
