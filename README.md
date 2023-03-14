```bash
fastestdet -i 3.jpeg --param-path models/FastestDet.param --model-path models/FastestDet.bin --classes-path models/classes.toml -o out.jpg
export GST_PLUGIN_PATH_1_0=$(pwd)/target/debug
export GST_DEBUG=*:2,fastestdet:5
```
