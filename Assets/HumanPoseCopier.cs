using UnityEngine;
using UnityEngine.Animations;

public class HumanPoseCopier : MonoBehaviour
{
    public Animator source;  // humanoid
    public Animator target;  // humanoid
    private bool doOnce = true;
    private bool doTwice = false;

    HumanPoseHandler srcHandler, dstHandler;
    HumanPose pose;

    void Awake() {
        srcHandler = new HumanPoseHandler(source.avatar, source.transform);
        dstHandler = new HumanPoseHandler(target.avatar, target.transform);
        pose = new HumanPose();
    }

    void LateUpdate()
    {
        if (doOnce)
        {
            // Read source
            srcHandler.GetHumanPose(ref pose);

            // (Optional) keep target root fixed:
            // pose.bodyPosition = Vector3.zero; // or target.transform.InverseTransformPoint(target.transform.position)
            // pose.bodyRotation = Quaternion.identity;

            // Write to target
            dstHandler.SetHumanPose(ref pose);
            doOnce = false;
            doTwice = true;
        }
        else if (doTwice)
        {
                        // Read source
            srcHandler.GetHumanPose(ref pose);

            // (Optional) keep target root fixed:
            // pose.bodyPosition = Vector3.zero; // or target.transform.InverseTransformPoint(target.transform.position)
            // pose.bodyRotation = Quaternion.identity;

            // Write to target
            dstHandler.SetHumanPose(ref pose);
            target.enabled = false;
            doTwice = false;
        }
    }
}
