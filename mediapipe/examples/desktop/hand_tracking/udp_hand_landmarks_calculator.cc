// Copyright 2026 LongDex overlay (local MediaPipe tree).
//
// Sink calculator: WORLD_LANDMARKS (+ optional HANDEDNESS) → UDP datagram.
// Does not output a stream. demo_run_graph_main_gpu.cc can keep polling
// output_video only.
//
// Env (read in Open; change IP without rebuild):
//   LONGDEX_HAND_UDP_HOST   default 127.0.0.1
//   LONGDEX_HAND_UDP_PORT   default 59100
//   LONGDEX_HAND_UDP_JSON   default 0; "1" sends one JSON object per datagram
//   LONGDEX_HAND_UDP_FLAGS  default 1 (bit0 = image already mirrored)
//
// Binary layout (little-endian), magic "LDH1": see aigc_doc/HAND_UDP_PLAN.md.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace {

constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kMagic[4] = {'L', 'D', 'H', '1'};
constexpr uint16_t kVersion = 1;
constexpr int kMaxHands = 2;
constexpr int kLandmarksPerHand = 21;
constexpr uint16_t kDefaultPort = 59100;
constexpr int kSendFailLogEvery = 30;

std::string EnvOr(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || v[0] == '\0') {
    return std::string(fallback);
  }
  return std::string(v);
}

uint16_t EnvPort() {
  const std::string s = EnvOr("LONGDEX_HAND_UDP_PORT", "59100");
  char* end = nullptr;
  const long p = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || p <= 0 || p > 65535) {
    return kDefaultPort;
  }
  return static_cast<uint16_t>(p);
}

uint32_t EnvFlags() {
  const std::string s = EnvOr("LONGDEX_HAND_UDP_FLAGS", "1");
  char* end = nullptr;
  const unsigned long f = std::strtoul(s.c_str(), &end, 10);
  if (end == s.c_str()) {
    return 1;
  }
  return static_cast<uint32_t>(f);
}

bool EnvJson() {
  const std::string s = EnvOr("LONGDEX_HAND_UDP_JSON", "0");
  return s == "1" || s == "true" || s == "TRUE" || s == "yes";
}

uint64_t UnixNs() {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    return 0;
  }
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

void PutU8(std::vector<uint8_t>* b, uint8_t v) { b->push_back(v); }

void PutU16(std::vector<uint8_t>* b, uint16_t v) {
  b->push_back(static_cast<uint8_t>(v & 0xff));
  b->push_back(static_cast<uint8_t>((v >> 8) & 0xff));
}

void PutU32(std::vector<uint8_t>* b, uint32_t v) {
  b->push_back(static_cast<uint8_t>(v & 0xff));
  b->push_back(static_cast<uint8_t>((v >> 8) & 0xff));
  b->push_back(static_cast<uint8_t>((v >> 16) & 0xff));
  b->push_back(static_cast<uint8_t>((v >> 24) & 0xff));
}

void PutU64(std::vector<uint8_t>* b, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    b->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
  }
}

void PutF32(std::vector<uint8_t>* b, float v) {
  uint32_t bits = 0;
  static_assert(sizeof(float) == 4, "float must be 32-bit");
  std::memcpy(&bits, &v, 4);
  PutU32(b, bits);
}

enum Side : uint8_t { kSideLeft = 0, kSideRight = 1, kSideUnknown = 255 };

Side SideFromLabel(const std::string& label) {
  if (label == "Left" || label == "left" || label == "LEFT") {
    return kSideLeft;
  }
  if (label == "Right" || label == "right" || label == "RIGHT") {
    return kSideRight;
  }
  return kSideUnknown;
}

const char* SideName(uint8_t side) {
  if (side == kSideLeft) {
    return "left";
  }
  if (side == kSideRight) {
    return "right";
  }
  return "unknown";
}

struct HandPose {
  uint8_t side = kSideUnknown;
  float score = 1.0f;
  float xyz[kLandmarksPerHand][3] = {};
};

void FillHand(const LandmarkList& lm, HandPose* out) {
  const int n = std::min(lm.landmark_size(), kLandmarksPerHand);
  for (int i = 0; i < n; ++i) {
    out->xyz[i][0] = lm.landmark(i).x();
    out->xyz[i][1] = lm.landmark(i).y();
    out->xyz[i][2] = lm.landmark(i).z();
  }
}

void ApplyHandedness(const std::vector<ClassificationList>* handedness, int i,
                     HandPose* out) {
  if (handedness == nullptr || i < 0 ||
      i >= static_cast<int>(handedness->size())) {
    return;
  }
  const ClassificationList& cl = (*handedness)[i];
  if (cl.classification_size() <= 0) {
    return;
  }
  const Classification& c = cl.classification(0);
  out->side = SideFromLabel(c.label());
  if (c.has_score()) {
    out->score = c.score();
  }
}

std::string JsonEscape(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o.push_back('\\');
    }
    o.push_back(c);
  }
  return o;
}

std::string ToJson(uint32_t seq, uint64_t t_ns, uint32_t flags,
                   const std::vector<HandPose>& hands) {
  std::string s;
  s.reserve(2048);
  s += "{\"v\":1,\"seq\":";
  s += std::to_string(seq);
  s += ",\"t_unix_ns\":";
  s += std::to_string(t_ns);
  s += ",\"flags\":";
  s += std::to_string(flags);
  s += ",\"hands\":[";
  for (size_t h = 0; h < hands.size(); ++h) {
    if (h) {
      s += ',';
    }
    s += "{\"side\":\"";
    s += JsonEscape(SideName(hands[h].side));
    s += "\",\"score\":";
    s += std::to_string(hands[h].score);
    s += ",\"xyz\":[";
    for (int i = 0; i < kLandmarksPerHand; ++i) {
      if (i) {
        s += ',';
      }
      s += '[';
      s += std::to_string(hands[h].xyz[i][0]);
      s += ',';
      s += std::to_string(hands[h].xyz[i][1]);
      s += ',';
      s += std::to_string(hands[h].xyz[i][2]);
      s += ']';
    }
    s += "]}";
  }
  s += "]}";
  return s;
}

std::vector<uint8_t> ToBinary(uint32_t seq, uint64_t t_ns, uint32_t flags,
                              const std::vector<HandPose>& hands) {
  std::vector<uint8_t> b;
  b.reserve(24 + hands.size() * 260);
  b.insert(b.end(), kMagic, kMagic + 4);
  PutU16(&b, kVersion);
  PutU16(&b, static_cast<uint16_t>(hands.size()));
  PutU32(&b, seq);
  PutU64(&b, t_ns);
  PutU32(&b, flags);
  for (const HandPose& h : hands) {
    PutU8(&b, h.side);
    PutU8(&b, 0);
    PutU8(&b, 0);
    PutU8(&b, 0);
    PutF32(&b, h.score);
    for (int i = 0; i < kLandmarksPerHand; ++i) {
      PutF32(&b, h.xyz[i][0]);
      PutF32(&b, h.xyz[i][1]);
      PutF32(&b, h.xyz[i][2]);
    }
  }
  return b;
}

}  // namespace

class UdpHandLandmarksCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kWorldLandmarksTag));
    cc->Inputs().Tag(kWorldLandmarksTag).Set<std::vector<LandmarkList>>();
    if (cc->Inputs().HasTag(kHandednessTag)) {
      cc->Inputs().Tag(kHandednessTag).Set<std::vector<ClassificationList>>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    host_ = EnvOr("LONGDEX_HAND_UDP_HOST", "127.0.0.1");
    port_ = EnvPort();
    flags_ = EnvFlags();
    json_ = EnvJson();

    sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ < 0) {
      return absl::InternalError(std::string("UDP socket: ") +
                                 std::strerror(errno));
    }
    const int fl = ::fcntl(sock_, F_GETFL, 0);
    if (fl >= 0) {
      ::fcntl(sock_, F_SETFL, fl | O_NONBLOCK);
    }

    std::memset(&addr_, 0, sizeof(addr_));
    addr_.sin_family = AF_INET;
    addr_.sin_port = htons(port_);
    if (::inet_pton(AF_INET, host_.c_str(), &addr_.sin_addr) != 1) {
      ::close(sock_);
      sock_ = -1;
      return absl::InvalidArgumentError(
          "LONGDEX_HAND_UDP_HOST must be IPv4 (got '" + host_ + "')");
    }

    ABSL_LOG(INFO) << "UdpHandLandmarksCalculator -> " << host_ << ":" << port_
                   << (json_ ? " json" : " binary LDH1");
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kWorldLandmarksTag).IsEmpty()) {
      return absl::OkStatus();
    }
    const auto& worlds =
        cc->Inputs().Tag(kWorldLandmarksTag).Get<std::vector<LandmarkList>>();
    if (worlds.empty()) {
      return absl::OkStatus();
    }

    const std::vector<ClassificationList>* handedness = nullptr;
    if (cc->Inputs().HasTag(kHandednessTag) &&
        !cc->Inputs().Tag(kHandednessTag).IsEmpty()) {
      handedness =
          &cc->Inputs().Tag(kHandednessTag).Get<std::vector<ClassificationList>>();
    }

    const int n = std::min(static_cast<int>(worlds.size()), kMaxHands);
    std::vector<HandPose> hands(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      FillHand(worlds[i], &hands[i]);
      ApplyHandedness(handedness, i, &hands[i]);
    }

    ++seq_;
    const uint64_t t_ns = UnixNs();
    if (json_) {
      const std::string payload = ToJson(seq_, t_ns, flags_, hands);
      Send(payload.data(), payload.size());
    } else {
      const std::vector<uint8_t> payload = ToBinary(seq_, t_ns, flags_, hands);
      Send(payload.data(), payload.size());
    }
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    (void)cc;
    if (sock_ >= 0) {
      ::close(sock_);
      sock_ = -1;
    }
    return absl::OkStatus();
  }

 private:
  void Send(const void* data, size_t n) {
    if (sock_ < 0 || n == 0) {
      return;
    }
    const ssize_t sent =
        ::sendto(sock_, data, n, 0, reinterpret_cast<sockaddr*>(&addr_),
                 sizeof(addr_));
    if (sent < 0) {
      ++send_fail_;
      if (send_fail_ == 1 || send_fail_ % kSendFailLogEvery == 0) {
        ABSL_LOG(WARNING) << "UDP sendto " << host_ << ":" << port_
                          << " failed x" << send_fail_ << ": "
                          << std::strerror(errno);
      }
    }
  }

  int sock_ = -1;
  sockaddr_in addr_{};
  std::string host_;
  uint16_t port_ = kDefaultPort;
  uint32_t flags_ = 1;
  bool json_ = false;
  uint32_t seq_ = 0;
  uint32_t send_fail_ = 0;
};

REGISTER_CALCULATOR(UdpHandLandmarksCalculator);

}  // namespace mediapipe
