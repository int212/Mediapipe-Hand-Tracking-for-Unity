using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SetLine : MonoBehaviour
{
    LineRenderer line;
    [Range(0, 1)] public float width;
    public Material lineMaterial;
    public Transform origin, destination;

    private void Start()
    {
        line = this.GetComponent<LineRenderer>();
        line.material = lineMaterial;
        line.startWidth = width;
        line.endWidth = width;
    }
    private void Update()
    {
        line.SetPosition(0, origin.localPosition);
        line.SetPosition(1, destination.localPosition);

    }
}
