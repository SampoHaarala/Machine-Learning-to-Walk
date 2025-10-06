using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Records the positions (and optionally rotations) of specific bones on a humanoid during a looping animation.
/// Attach this script to any GameObject in your scene and populate the bone fields via the Inspector.
/// When the recording duration elapses the collected data is saved to a JSON file in the persistent data path.
/// You can adjust the recordDuration to match the length of your animation loop (e.g. ~0.77 seconds for a 23‑frame loop at 30 FPS).
/// </summary>
public class AnimationDataRecorder : MonoBehaviour
{
    [Header("Transforms to record")]
    [Tooltip("The root hip bone of your character.")]
    public Transform hip;
    [Tooltip("Left leg bone.")]
    public Transform leftLeg;
    [Tooltip("Right leg bone.")]
    public Transform rightLeg;

    [Tooltip("Left knee bone.")]
    public Transform leftKnee;

    [Tooltip("Right knee bone.")]
    public Transform rightKnee;

    [Tooltip("Left ankle/foot bone.")]
    public Transform leftAnkle;

    [Tooltip("Right ankle/foot bone.")]
    public Transform rightAnkle;

    [Header("Recording settings")]
    [Tooltip("Duration in seconds to record. Should cover one full animation loop.")]
    public float recordDuration = 1f;

    [Tooltip("Sample interval in seconds. Use 0 to record every frame.")]
    public float sampleInterval = 0f;

    // Internal list to store frame data
    private readonly List<FrameData> _frames = new List<FrameData>();
    private float _elapsed;
    private float _nextSampleTime;
    private bool _isRecording;

    private void Start()
    {
        // Start recording on play
        _isRecording = true;
        _elapsed = 0f;
        _nextSampleTime = 0f;
        _frames.Clear();
    }

    private void Update()
    {
        if (!_isRecording)
        {
            return;
        }

        _elapsed += Time.deltaTime;

        // If sample interval is zero record every frame, otherwise sample when elapsed exceeds nextSampleTime
        if (sampleInterval <= 0f || _elapsed >= _nextSampleTime)
        {
            RecordFrame();
            if (sampleInterval > 0f)
            {
                _nextSampleTime += sampleInterval;
            }
        }

        // Stop recording when duration exceeded
        if (_elapsed >= recordDuration)
        {
            _isRecording = false;
            SaveToJson();
        }
    }

    /// <summary>
    /// Captures the current positions of the configured bones and stores them in the frames list.
    /// </summary>
    private void RecordFrame()
    {
        var frame = new FrameData
        {
            time = _elapsed
        };

        if (hip != null)
        {
            frame.positions["hip"] = hip.localPosition;
        }
        if (leftLeg != null)
        {
            frame.positions["leftLeg"] = leftLeg.localPosition;
        }
        if (leftLeg != null)
        {
            frame.positions["leftLeg"] = leftLeg.localPosition;
        }
        if (leftKnee != null)
        {
            frame.positions["leftKnee"] = leftKnee.localPosition;
        }
        if (rightKnee != null)
        {
            frame.positions["rightKnee"] = rightKnee.localPosition;
        }
        if (leftAnkle != null)
        {
            frame.positions["leftAnkle"] = leftAnkle.localPosition;
        }
        if (rightAnkle != null)
        {
            frame.positions["rightAnkle"] = rightAnkle.localPosition;
        }

        _frames.Add(frame);
    }

    /// <summary>
    /// Serializes the recorded frames to JSON and writes them to a file on disk.
    /// The output file will be created in Application.persistentDataPath.
    /// </summary>
    private void SaveToJson()
    {
        // Wrap the list in a container so Unity's JsonUtility can serialize it
        var container = new FrameDataContainer { frames = _frames };
        string json = JsonUtility.ToJson(container, prettyPrint: true);

        // Build a filename based on date/time for uniqueness
        string fileName = $"animation_data_{System.DateTime.Now:yyyyMMdd_HHmmss}.json";
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
        public Dictionary<string, Vector3> positions = new Dictionary<string, Vector3>();
    }

    /// <summary>
    /// Wrapper class so that JsonUtility can serialize a list of frames.
    /// </summary>
    [System.Serializable]
    private class FrameDataContainer
    {
        public List<FrameData> frames;
    }
}