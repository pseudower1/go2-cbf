// lidar_recon.cpp — which Go2 LiDAR product actually delivers over DDS?
// Subscribes to the raw cloud, the height map, and the compressed voxel map
// simultaneously and prints a per-topic message count + summary each second.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <unitree/robot/channel/channel_subscriber.hpp>

#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/go2/HeightMap_.hpp>
#include <unitree/idl/go2/VoxelMapCompressed_.hpp>
#include <unitree/idl/go2/LidarState_.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

using PointCloud2 = sensor_msgs::msg::dds_::PointCloud2_;
using HeightMap   = unitree_go::msg::dds_::HeightMap_;
using VoxelMap    = unitree_go::msg::dds_::VoxelMapCompressed_;
using LidarState  = unitree_go::msg::dds_::LidarState_;

struct Slot {
  std::atomic<uint64_t> count{0};
  std::mutex mtx;
  std::string summary{"(none)"};
};
static Slot g_cloud, g_height, g_voxel, g_state;

static void set_sum(Slot& s, const std::string& str) {
  std::lock_guard<std::mutex> lk(s.mtx);
  s.summary = str;
}

static void on_cloud(const void* p) {
  const auto& m = *static_cast<const PointCloud2*>(p);
  const uint32_t step = m.point_step();
  const size_t n = step ? m.data().size() / step : 0;
  std::ostringstream ss;
  ss << "PointCloud2 pts=" << n << " bytes=" << m.data().size()
     << " frame='" << m.header().frame_id() << "' step=" << step;
  set_sum(g_cloud, ss.str()); g_cloud.count.fetch_add(1);
}
static void on_height(const void* p) {
  const auto& m = *static_cast<const HeightMap*>(p);
  size_t occ = 0; float zmin = 1e9f, zmax = -1e9f;
  for (float v : m.data()) {
    if (!std::isfinite(v)) continue;
    zmin = std::min(zmin, v); zmax = std::max(zmax, v);
    if (v > -0.05f) ++occ;            // crude "above floor" count
  }
  std::ostringstream ss;
  ss << "HeightMap " << m.width() << "x" << m.height() << " res=" << m.resolution()
     << "m origin(" << m.origin()[0] << "," << m.origin()[1] << ")"
     << " frame='" << m.frame_id() << "' cells=" << m.data().size()
     << " finite_z[" << (zmin>1e8f?0:zmin) << "," << (zmax<-1e8f?0:zmax) << "]"
     << " above_floor=" << occ;
  set_sum(g_height, ss.str()); g_height.count.fetch_add(1);
}
static void on_voxel(const void* p) {
  const auto& m = *static_cast<const VoxelMap*>(p);
  std::ostringstream ss;
  ss << "VoxelMapCompressed " << m.width()[0] << "x" << m.width()[1] << "x" << m.width()[2]
     << " res=" << m.resolution() << "m src_size=" << m.src_size()
     << " comp_bytes=" << m.data().size() << " frame='" << m.frame_id() << "'";
  set_sum(g_voxel, ss.str()); g_voxel.count.fetch_add(1);
}
static void on_state(const void* p) {
  const auto& s = *static_cast<const LidarState*>(p);
  std::ostringstream ss;
  ss << "rpm=" << s.sys_rotation_speed() << " cloud_hz=" << s.cloud_frequency()
     << " size=" << s.cloud_size() << " err=" << int(s.error_state())
     << " dirty=" << int(s.dirty_percentage()) << "%";
  set_sum(g_state, ss.str()); g_state.count.fetch_add(1);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: " << argv[0] << " <iface|lo>\n"; return 1; }
  std::cout << std::unitbuf;
  ChannelFactory::Instance()->Init(0, argv[1]);
  std::cout << "[recon] DDS up on '" << argv[1] << "'\n";

  ChannelSubscriberPtr<PointCloud2> a(new ChannelSubscriber<PointCloud2>("rt/utlidar/cloud"));
  a->InitChannel(on_cloud, 1);
  ChannelSubscriberPtr<HeightMap> b(new ChannelSubscriber<HeightMap>("rt/utlidar/height_map_array"));
  b->InitChannel(on_height, 1);
  ChannelSubscriberPtr<VoxelMap> c(new ChannelSubscriber<VoxelMap>("rt/utlidar/voxel_map_compressed"));
  c->InitChannel(on_voxel, 1);
  ChannelSubscriberPtr<LidarState> d(new ChannelSubscriber<LidarState>("rt/utlidar/lidar_state"));
  d->InitChannel(on_state, 1);

  std::cout << "[recon] subscribed: rt/utlidar/{cloud, height_map_array, voxel_map_compressed, lidar_state}\n\n";

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto line = [](const char* name, Slot& s) {
      std::lock_guard<std::mutex> lk(s.mtx);
      std::cout << "  " << name << " n=" << s.count.load() << "  " << s.summary << "\n";
    };
    std::cout << "---- counts ----\n";
    line("cloud      ", g_cloud);
    line("height_map ", g_height);
    line("voxel_map  ", g_voxel);
    line("lidar_state", g_state);
  }
  return 0;
}
