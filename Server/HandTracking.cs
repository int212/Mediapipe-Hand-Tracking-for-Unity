using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandTracking : MonoBehaviour
{
    public Transform[] points;
    public float smooth = 1.4f;
    Vector3 velocity = Vector3.zero;
    private void FixedUpdate()
    {
        string data = ReceiveByUDP._instance.data;
        if (data.Length == 0) return;
        data = data.Remove(0, 1);
        data = data.Remove(data.Length - 1, 1);
        
        string[] strs = data.Split(',');


        for (int i = 0; i < 21; i++)
        {
            float x = 5 - float.Parse(strs[i * 3]) / 100;
            float y = float.Parse(strs[i * 3 + 1]) / 100;
            float z = 0 - float.Parse(strs[i * 3 + 2]) / 30;

            //points[i].transform.localPosition = new Vector3(x, y, z);
            points[i].transform.localPosition = Vector3.SmoothDamp(points[i].transform.position, new Vector3(x, y, z), ref velocity, Time.deltaTime * smooth);
        }

    }
}
