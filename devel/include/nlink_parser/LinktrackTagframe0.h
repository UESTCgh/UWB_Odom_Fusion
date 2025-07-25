// Generated by gencpp from file nlink_parser/LinktrackTagframe0.msg
// DO NOT EDIT!


#ifndef NLINK_PARSER_MESSAGE_LINKTRACKTAGFRAME0_H
#define NLINK_PARSER_MESSAGE_LINKTRACKTAGFRAME0_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace nlink_parser
{
template <class ContainerAllocator>
struct LinktrackTagframe0_
{
  typedef LinktrackTagframe0_<ContainerAllocator> Type;

  LinktrackTagframe0_()
    : role(0)
    , id(0)
    , local_time(0)
    , system_time(0)
    , voltage(0.0)
    , pos_3d()
    , eop_3d()
    , vel_3d()
    , dis_arr()
    , angle_3d()
    , quaternion()
    , imu_gyro_3d()
    , imu_acc_3d()  {
      pos_3d.assign(0.0);

      eop_3d.assign(0.0);

      vel_3d.assign(0.0);

      dis_arr.assign(0.0);

      angle_3d.assign(0.0);

      quaternion.assign(0.0);

      imu_gyro_3d.assign(0.0);

      imu_acc_3d.assign(0.0);
  }
  LinktrackTagframe0_(const ContainerAllocator& _alloc)
    : role(0)
    , id(0)
    , local_time(0)
    , system_time(0)
    , voltage(0.0)
    , pos_3d()
    , eop_3d()
    , vel_3d()
    , dis_arr()
    , angle_3d()
    , quaternion()
    , imu_gyro_3d()
    , imu_acc_3d()  {
  (void)_alloc;
      pos_3d.assign(0.0);

      eop_3d.assign(0.0);

      vel_3d.assign(0.0);

      dis_arr.assign(0.0);

      angle_3d.assign(0.0);

      quaternion.assign(0.0);

      imu_gyro_3d.assign(0.0);

      imu_acc_3d.assign(0.0);
  }



   typedef uint8_t _role_type;
  _role_type role;

   typedef uint8_t _id_type;
  _id_type id;

   typedef uint32_t _local_time_type;
  _local_time_type local_time;

   typedef uint32_t _system_time_type;
  _system_time_type system_time;

   typedef float _voltage_type;
  _voltage_type voltage;

   typedef boost::array<float, 3>  _pos_3d_type;
  _pos_3d_type pos_3d;

   typedef boost::array<float, 3>  _eop_3d_type;
  _eop_3d_type eop_3d;

   typedef boost::array<float, 3>  _vel_3d_type;
  _vel_3d_type vel_3d;

   typedef boost::array<float, 8>  _dis_arr_type;
  _dis_arr_type dis_arr;

   typedef boost::array<float, 3>  _angle_3d_type;
  _angle_3d_type angle_3d;

   typedef boost::array<float, 4>  _quaternion_type;
  _quaternion_type quaternion;

   typedef boost::array<float, 3>  _imu_gyro_3d_type;
  _imu_gyro_3d_type imu_gyro_3d;

   typedef boost::array<float, 3>  _imu_acc_3d_type;
  _imu_acc_3d_type imu_acc_3d;





  typedef boost::shared_ptr< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> const> ConstPtr;

}; // struct LinktrackTagframe0_

typedef ::nlink_parser::LinktrackTagframe0_<std::allocator<void> > LinktrackTagframe0;

typedef boost::shared_ptr< ::nlink_parser::LinktrackTagframe0 > LinktrackTagframe0Ptr;
typedef boost::shared_ptr< ::nlink_parser::LinktrackTagframe0 const> LinktrackTagframe0ConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator1> & lhs, const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator2> & rhs)
{
  return lhs.role == rhs.role &&
    lhs.id == rhs.id &&
    lhs.local_time == rhs.local_time &&
    lhs.system_time == rhs.system_time &&
    lhs.voltage == rhs.voltage &&
    lhs.pos_3d == rhs.pos_3d &&
    lhs.eop_3d == rhs.eop_3d &&
    lhs.vel_3d == rhs.vel_3d &&
    lhs.dis_arr == rhs.dis_arr &&
    lhs.angle_3d == rhs.angle_3d &&
    lhs.quaternion == rhs.quaternion &&
    lhs.imu_gyro_3d == rhs.imu_gyro_3d &&
    lhs.imu_acc_3d == rhs.imu_acc_3d;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator1> & lhs, const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace nlink_parser

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >
{
  static const char* value()
  {
    return "20cc09884b3e1aa830a1d8a71796a857";
  }

  static const char* value(const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x20cc09884b3e1aa8ULL;
  static const uint64_t static_value2 = 0x30a1d8a71796a857ULL;
};

template<class ContainerAllocator>
struct DataType< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >
{
  static const char* value()
  {
    return "nlink_parser/LinktrackTagframe0";
  }

  static const char* value(const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 role\n"
"uint8 id\n"
"uint32 local_time\n"
"uint32 system_time\n"
"float32 voltage\n"
"float32[3] pos_3d\n"
"float32[3] eop_3d\n"
"float32[3] vel_3d\n"
"float32[8] dis_arr\n"
"float32[3] angle_3d\n"
"float32[4] quaternion\n"
"float32[3] imu_gyro_3d\n"
"float32[3] imu_acc_3d\n"
;
  }

  static const char* value(const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.role);
      stream.next(m.id);
      stream.next(m.local_time);
      stream.next(m.system_time);
      stream.next(m.voltage);
      stream.next(m.pos_3d);
      stream.next(m.eop_3d);
      stream.next(m.vel_3d);
      stream.next(m.dis_arr);
      stream.next(m.angle_3d);
      stream.next(m.quaternion);
      stream.next(m.imu_gyro_3d);
      stream.next(m.imu_acc_3d);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct LinktrackTagframe0_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::nlink_parser::LinktrackTagframe0_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::nlink_parser::LinktrackTagframe0_<ContainerAllocator>& v)
  {
    if (false || !indent.empty())
      s << std::endl;
    s << indent << "role: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.role);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "id: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.id);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "local_time: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.local_time);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "system_time: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.system_time);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "voltage: ";
    Printer<float>::stream(s, indent + "  ", v.voltage);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "pos_3d: ";
    if (v.pos_3d.empty() || true)
      s << "[";
    for (size_t i = 0; i < v.pos_3d.size(); ++i)
    {
      if (true && i > 0)
        s << ", ";
      else if (!true)
        s << std::endl << indent << "  -";
      Printer<float>::stream(s, true ? std::string() : indent + "    ", v.pos_3d[i]);
    }
    if (v.pos_3d.empty() || true)
      s << "]";
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "eop_3d: ";
    if (v.eop_3d.empty() || true)
      s << "[";
    for (size_t i = 0; i < v.eop_3d.size(); ++i)
    {
      if (true && i > 0)
        s << ", ";
      else if (!true)
        s << std::endl << indent << "  -";
      Printer<float>::stream(s, true ? std::string() : indent + "    ", v.eop_3d[i]);
    }
    if (v.eop_3d.empty() || true)
      s << "]";
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "vel_3d: ";
    if (v.vel_3d.empty() || true)
      s << "[";
    for (size_t i = 0; i < v.vel_3d.size(); ++i)
    {
      if (true && i > 0)
        s << ", ";
      else if (!true)
        s << std::endl << indent << "  -";
      Printer<float>::stream(s, true ? std::string() : indent + "    ", v.vel_3d[i]);
    }
    if (v.vel_3d.empty() || true)
      s << "]";
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "dis_arr: ";
    if (v.dis_arr.empty() || true)
      s << "[";
    for (size_t i = 0; i < v.dis_arr.size(); ++i)
    {
      if (true && i > 0)
        s << ", ";
      else if (!true)
        s << std::endl << indent << "  -";
      Printer<float>::stream(s, true ? std::string() : indent + "    ", v.dis_arr[i]);
    }
    if (v.dis_arr.empty() || true)
      s << "]";
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "angle_3d: ";
    if (v.angle_3d.empty() || true)
      s << "[";
    for (size_t i = 0; i < v.angle_3d.size(); ++i)
    {
      if (true && i > 0)
        s << ", ";
      else if (!true)
        s << std::endl << indent << "  -";
      Printer<float>::stream(s, true ? std::string() : indent + "    ", v.angle_3d[i]);
    }
    if (v.angle_3d.empty() || true)
      s << "]";
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "quaternion: ";
    if (v.quaternion.empty() || true)
      s << "[";
    for (size_t i = 0; i < v.quaternion.size(); ++i)
    {
      if (true && i > 0)
        s << ", ";
      else if (!true)
        s << std::endl << indent << "  -";
      Printer<float>::stream(s, true ? std::string() : indent + "    ", v.quaternion[i]);
    }
    if (v.quaternion.empty() || true)
      s << "]";
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "imu_gyro_3d: ";
    if (v.imu_gyro_3d.empty() || true)
      s << "[";
    for (size_t i = 0; i < v.imu_gyro_3d.size(); ++i)
    {
      if (true && i > 0)
        s << ", ";
      else if (!true)
        s << std::endl << indent << "  -";
      Printer<float>::stream(s, true ? std::string() : indent + "    ", v.imu_gyro_3d[i]);
    }
    if (v.imu_gyro_3d.empty() || true)
      s << "]";
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "imu_acc_3d: ";
    if (v.imu_acc_3d.empty() || true)
      s << "[";
    for (size_t i = 0; i < v.imu_acc_3d.size(); ++i)
    {
      if (true && i > 0)
        s << ", ";
      else if (!true)
        s << std::endl << indent << "  -";
      Printer<float>::stream(s, true ? std::string() : indent + "    ", v.imu_acc_3d[i]);
    }
    if (v.imu_acc_3d.empty() || true)
      s << "]";
  }
};

} // namespace message_operations
} // namespace ros

#endif // NLINK_PARSER_MESSAGE_LINKTRACKTAGFRAME0_H
