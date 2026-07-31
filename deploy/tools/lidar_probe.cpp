// lidar_probe.cpp — minimal standalone reader for the Go2 L1 (head) LiDAR.
//
// Purpose: prove we can actually receive the LiDAR point cloud over the Unitree
// SDK2 DDS channels (the SAME mechanism the deployer uses for rt/lowstate), and
// print enough about it to start Phase D (cluster the cloud into obstacle
// circles -> feed the MPPI-CBF planner).
//
// Topics (Go2 L1 / "utlidar" service — runs independently of the sport service,
// so it keeps publishing even in low-level RL control mode, unlike sportmodestate):
//   rt/utlidar/cloud        sensor_msgs/PointCloud2   the point cloud
//   rt/utlidar/lidar_state  unitree_go/LidarState     rpm / freq / error / dirty
//   rt/utlidar/switch       std_msgs/String "ON"/"OFF"  enable/standby the lidar
//
// Build:  see tools/build_lidar_probe.sh  (no Torch needed — links only the SDK)
// Run:    ./lidar_probe <iface>     e.g.  ./lidar_probe enp128s31f6
//         ./lidar_probe lo          (sim/loopback)
//         ./lidar_probe <iface> --off   (put lidar in standby and exit)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include <unitree/idl/go2/LidarState_.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

using PointCloud2 = sensor_msgs::msg::dds_::PointCloud2_;
using PointField  = sensor_msgs::msg::dds_::PointField_;
using StringMsg   = std_msgs::msg::dds_::String_;
using LidarState  = unitree_go::msg::dds_::LidarState_;

static constexpr char TOPIC_CLOUD[]  = "rt/utlidar/cloud";
static constexpr char TOPIC_STATE[]  = "rt/utlidar/lidar_state";
static constexpr char TOPIC_SWITCH[] = "rt/utlidar/switch";

// ---- shared state updated by the DDS callbacks --------------------------------
static std::mutex g_mtx;
static std::atomic<uint64_t> g_cloud_count{0};
static std::atomic<uint64_t> g_state_count{0};
static double g_last_cloud_t = 0.0;
static std::string g_summary = "(no cloud yet)";
static std::string g_state_str = "(no lidar_state yet)";

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Locate the byte offset of a named FLOAT32 field (x/y/z/intensity) in a point.
static int field_offset(const PointCloud2& m, const std::string& name) {
  for (const auto& f : m.fields()) {
    if (f.name() == name) return static_cast<int>(f.offset());
  }
  return -1;
}

static void on_cloud(const void* msgptr) {
  const PointCloud2& m = *static_cast<const PointCloud2*>(msgptr);
  const uint32_t step = m.point_step();
  const size_t n_pts = step ? m.data().size() / step : 0;

  const int ox = field_offset(m, "x");
  const int oy = field_offset(m, "y");
  const int oz = field_offset(m, "z");

  // Scan: overall bounding box + nearest point in a forward "obstacle" box.
  // (Points are in the LiDAR's own frame; this is a raw sanity check, not yet
  //  transformed into the robot base frame — that comes during integration.)
  size_t valid = 0;
  float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f, minz = 1e9f, maxz = -1e9f;
  float nearest = std::numeric_limits<float>::infinity();
  float near_xyz[3] = {0, 0, 0};

  if (ox >= 0 && oy >= 0 && oz >= 0 && n_pts) {
    const uint8_t* base = m.data().data();
    for (size_t i = 0; i < n_pts; ++i) {
      const uint8_t* p = base + i * step;
      float x, y, z;
      std::memcpy(&x, p + ox, 4);
      std::memcpy(&y, p + oy, 4);
      std::memcpy(&z, p + oz, 4);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
      if (x == 0.0f && y == 0.0f && z == 0.0f) continue;  // padding/no-return
      ++valid;
      minx = std::min(minx, x); maxx = std::max(maxx, x);
      miny = std::min(miny, y); maxy = std::max(maxy, y);
      minz = std::min(minz, z); maxz = std::max(maxz, z);
      // Forward box ~ in front of the robot, roughly chest height band.
      if (x > 0.15f && x < 4.0f && std::fabs(y) < 1.2f && z > -0.4f && z < 0.6f) {
        const float d = std::sqrt(x * x + y * y);
        if (d < nearest) { nearest = d; near_xyz[0] = x; near_xyz[1] = y; near_xyz[2] = z; }
      }
    }
  }

  std::ostringstream ss;
  ss << "pts=" << n_pts << " valid=" << valid
     << " frame='" << m.header().frame_id() << "'"
     << " step=" << step << "B  fields=[";
  for (size_t i = 0; i < m.fields().size(); ++i)
    ss << (i ? "," : "") << m.fields()[i].name();
  ss << "]";
  if (ox < 0 || oy < 0 || oz < 0) {
    ss << "  !! no x/y/z FLOAT32 fields found";
  } else if (valid) {
    ss << "\n      bbox x[" << minx << "," << maxx << "] y[" << miny << ","
       << maxy << "] z[" << minz << "," << maxz << "]";
    if (std::isfinite(nearest))
      ss << "\n      nearest-in-front: " << nearest << " m  at ("
         << near_xyz[0] << "," << near_xyz[1] << "," << near_xyz[2] << ")";
    else
      ss << "\n      nearest-in-front: (none in box — clear path)";
  }

  std::lock_guard<std::mutex> lk(g_mtx);
  g_summary = ss.str();
  g_last_cloud_t = now_s();
  g_cloud_count.fetch_add(1);
}

static void on_state(const void* msgptr) {
  const LidarState& s = *static_cast<const LidarState*>(msgptr);
  std::ostringstream ss;
  ss << "rpm(sys/com)=" << s.sys_rotation_speed() << "/" << s.com_rotation_speed()
     << " cloud_hz=" << s.cloud_frequency() << " size=" << s.cloud_size()
     << " err=" << int(s.error_state()) << " dirty=" << int(s.dirty_percentage()) << "%"
     << " fw=" << s.firmware_version();
  std::lock_guard<std::mutex> lk(g_mtx);
  g_state_str = ss.str();
  g_state_count.fetch_add(1);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <iface|lo> [--off]\n";
    return 1;
  }
  const std::string iface = argv[1];
  const bool turn_off = (argc > 2 && std::string(argv[2]) == "--off");

  std::cout << std::unitbuf;  // flush every line so live output isn't lost when piped/tee'd

  ChannelFactory::Instance()->Init(0, iface);
  std::cout << "[lidar_probe] DDS up on iface '" << iface << "'\n";

  // The L1 spins by default; publish ON to be sure (or OFF to standby & exit).
  ChannelPublisherPtr<StringMsg> sw(new ChannelPublisher<StringMsg>(TOPIC_SWITCH));
  sw->InitChannel();
  StringMsg cmd;
  cmd.data(turn_off ? "OFF" : "ON");
  sw->Write(cmd);
  std::cout << "[lidar_probe] sent switch '" << cmd.data() << "' on " << TOPIC_SWITCH << "\n";
  if (turn_off) { std::this_thread::sleep_for(std::chrono::milliseconds(300)); return 0; }

  ChannelSubscriberPtr<PointCloud2> cloud_sub(new ChannelSubscriber<PointCloud2>(TOPIC_CLOUD));
  cloud_sub->InitChannel(on_cloud, 1);
  ChannelSubscriberPtr<LidarState> state_sub(new ChannelSubscriber<LidarState>(TOPIC_STATE));
  state_sub->InitChannel(on_state, 1);
  std::cout << "[lidar_probe] subscribed to " << TOPIC_CLOUD << " and " << TOPIC_STATE
            << " — waiting for data...\n\n";

  uint64_t last_cloud = 0;
  double t_prev = now_s();
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const uint64_t c = g_cloud_count.load();
    const double t = now_s();
    const double hz = (c - last_cloud) / (t - t_prev);
    last_cloud = c; t_prev = t;

    std::string summary, state;
    double age;
    { std::lock_guard<std::mutex> lk(g_mtx);
      summary = g_summary; state = g_state_str; age = t - g_last_cloud_t; }

    if (c == 0) {
      std::cout << "[lidar_probe] no cloud yet ("
                << g_state_count.load() << " state msgs). "
                << "Check: lidar ON? right iface? robot powered?\n";
      std::cout << "             state: " << state << "\n";
    } else {
      std::cout << "[cloud ~" << hz << " Hz, age " << age << "s] " << summary << "\n";
      std::cout << "             state: " << state << "\n";
    }
  }
  return 0;
}
