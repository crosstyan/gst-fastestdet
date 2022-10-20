export GST_PLUGIN_PATH_1_0=$(pwd)/gst-fastestdet/target/debug
export GST_DEBUG=*:2,fastestdet:5

```bash
mosquitto_pub -h 192.168.1.11 -p 1883 -t hello/rumqtt -m join -d
mosquitto_pub -h 192.168.1.11 -p 1883 -t hello/rumqtt -m leave -d
```
