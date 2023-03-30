```bash
./target/debug/fastestdet -i 3.jpeg --param-path models/FastestDet.param --model-path models/FastestDet.bin --classes-path models/classes.toml -o out.jpg
./target/debug/fastestdet -i 3.jpeg --param-path models/yolo-fastestv2-opt.param --model-path models/yolo-fastestv2-opt.bin --classes-path models/classes.toml -o out.jpg
lldb-15 ./target/debug/fastestdet -- -i 3.jpeg --param-path models/yolo-fastestv2-opt.param --model-path models/yolo-fastestv2-opt.bin --classes-path models/classes.toml -o out.jpg
fastestdet -i 3.jpeg --param-path models/FastestDet.param --model-path models/FastestDet.bin --classes-path models/classes.toml -o out.jpg
export GST_PLUGIN_PATH_1_0=$(pwd)/target/debug
export GST_DEBUG=*:2,fastestdet:5
```

Find compiled `libncnn.so`. Use release build.

```bash
find . | grep libncnn.so
```
