# longdex-mediapipe

google/mediapipe 快照 + LongDex 手部关键点 UDP 发送端（私有协作仓库）。

## 来源与约定

- 快照自 [google/mediapipe](https://github.com/google/mediapipe)（Apache-2.0，保留 LICENSE），以单提交导入后在其上开发。
- `upstream` remote 指向 google/mediapipe；`origin` 指向本仓库。
- 不要提交 Bazel 输出（`bazel-*`）、`MODULE.bazel.lock`、模型/构建缓存。
- 协作走 feature 分支 + PR，不要直接推 main。

## 本地改动

### 1. 桌面示例编译修复（Ubuntu 24.04）

- `mediapipe/examples/desktop/BUILD`：`//third_party/gloop/base:init_google` → `//third_party:glog`（`third_party/gloop` 在上游导出中不存在）
- 三个 run_graph main 各补一行 `#include "mediapipe/framework/port/logging.h"`
- `third_party/opencv_linux.BUILD`：移除误入 BUILD 文件的 `new_local_repository` workspace 块，加 `visibility = public`

### 2. UDP world landmarks 发送端

- `mediapipe/examples/desktop/hand_tracking/udp_hand_landmarks_calculator.cc`
  `UdpHandLandmarksCalculator`：把 `HandLandmarkTrackingGpu` 的 `WORLD_LANDMARKS`（米制、原点在手 bbox 中心、21 点）+ `HANDEDNESS` 打成小端二进制 `LDH1` 后 UDP `sendto`。无手不发包；socket 非阻塞；目标地址用环境变量，改 IP 不用重编。
- `mediapipe/graphs/hand_tracking/hand_tracking_desktop_live_gpu_udp.pbtxt`：桌面 GPU 实时图新加 sink 节点（原图未动）。
- `mediapipe/examples/desktop/hand_tracking/BUILD`：calculator 的 `cc_library`（`alwayslink=1`）挂进 `hand_tracking_gpu`。

## 编译

```bash
export HERMETIC_PYTHON_VERSION=3.12
bazel build -c opt \
  --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
  //mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu
```

## 运行发送端

```bash
export LONGDEX_HAND_UDP_HOST=127.0.0.1   # 默认 127.0.0.1；跨机填接收方 IPv4
export LONGDEX_HAND_UDP_PORT=59100      # 默认 59100
export LONGDEX_HAND_UDP_JSON=0          # =1 改发 JSON（仅调试）
bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live_gpu_udp.pbtxt
```

## LDH1 载荷（小端）

```
magic "LDH1"(4) + version u16(=1) + n_hands u16 + seq u32 + t_unix_ns u64 + flags u32
+ 每只手 260 B：side u8(0左/1右/255未知) + pad3 + score f32 + xyz f32[21][3]
```

接收端（Python）在 LongDex-B1K 仓库 `longdex/teleop/vision/` 另行维护。
