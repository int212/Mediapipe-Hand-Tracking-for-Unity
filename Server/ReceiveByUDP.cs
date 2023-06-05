using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System;
using System.Text;

public class ReceiveByUDP : MonoBehaviour
{
    public static ReceiveByUDP _instance;

    Thread thread;
    UdpClient client;
    public int port = 9090;
    public bool startReceiveing = true;
    public string data;


    private void Awake()
    {
        _instance = this;
    }
    private void Start()
    {
        thread = new Thread(new ThreadStart(ReceiveData));
        thread.IsBackground = true;
        thread.Start();
    }
    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (startReceiveing)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] dataByte = client.Receive(ref anyIP);
                data = Encoding.UTF8.GetString(dataByte);
            }
            catch (System.Exception e)
            {

                Debug.Log(e.Message);
            }
        }
    }
}
