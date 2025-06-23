// Auto-generated. Do not edit!

// (in-package nlink_parser.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class IotFrame0Node {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.uid = null;
      this.dis = null;
      this.aoa_angle_horizontal = null;
      this.aoa_angle_vertical = null;
      this.fp_rssi = null;
      this.rx_rssi = null;
      this.user_data = null;
    }
    else {
      if (initObj.hasOwnProperty('uid')) {
        this.uid = initObj.uid
      }
      else {
        this.uid = 0;
      }
      if (initObj.hasOwnProperty('dis')) {
        this.dis = initObj.dis
      }
      else {
        this.dis = 0.0;
      }
      if (initObj.hasOwnProperty('aoa_angle_horizontal')) {
        this.aoa_angle_horizontal = initObj.aoa_angle_horizontal
      }
      else {
        this.aoa_angle_horizontal = 0.0;
      }
      if (initObj.hasOwnProperty('aoa_angle_vertical')) {
        this.aoa_angle_vertical = initObj.aoa_angle_vertical
      }
      else {
        this.aoa_angle_vertical = 0.0;
      }
      if (initObj.hasOwnProperty('fp_rssi')) {
        this.fp_rssi = initObj.fp_rssi
      }
      else {
        this.fp_rssi = 0.0;
      }
      if (initObj.hasOwnProperty('rx_rssi')) {
        this.rx_rssi = initObj.rx_rssi
      }
      else {
        this.rx_rssi = 0.0;
      }
      if (initObj.hasOwnProperty('user_data')) {
        this.user_data = initObj.user_data
      }
      else {
        this.user_data = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type IotFrame0Node
    // Serialize message field [uid]
    bufferOffset = _serializer.uint32(obj.uid, buffer, bufferOffset);
    // Serialize message field [dis]
    bufferOffset = _serializer.float32(obj.dis, buffer, bufferOffset);
    // Serialize message field [aoa_angle_horizontal]
    bufferOffset = _serializer.float32(obj.aoa_angle_horizontal, buffer, bufferOffset);
    // Serialize message field [aoa_angle_vertical]
    bufferOffset = _serializer.float32(obj.aoa_angle_vertical, buffer, bufferOffset);
    // Serialize message field [fp_rssi]
    bufferOffset = _serializer.float32(obj.fp_rssi, buffer, bufferOffset);
    // Serialize message field [rx_rssi]
    bufferOffset = _serializer.float32(obj.rx_rssi, buffer, bufferOffset);
    // Serialize message field [user_data]
    bufferOffset = _serializer.string(obj.user_data, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type IotFrame0Node
    let len;
    let data = new IotFrame0Node(null);
    // Deserialize message field [uid]
    data.uid = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [dis]
    data.dis = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [aoa_angle_horizontal]
    data.aoa_angle_horizontal = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [aoa_angle_vertical]
    data.aoa_angle_vertical = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [fp_rssi]
    data.fp_rssi = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rx_rssi]
    data.rx_rssi = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [user_data]
    data.user_data = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.user_data);
    return length + 28;
  }

  static datatype() {
    // Returns string type for a message object
    return 'nlink_parser/IotFrame0Node';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'df423b5fa843bedeae14cfe493476452';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint32 uid
    float32 dis
    float32 aoa_angle_horizontal
    float32 aoa_angle_vertical
    float32 fp_rssi
    float32 rx_rssi
    string user_data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new IotFrame0Node(null);
    if (msg.uid !== undefined) {
      resolved.uid = msg.uid;
    }
    else {
      resolved.uid = 0
    }

    if (msg.dis !== undefined) {
      resolved.dis = msg.dis;
    }
    else {
      resolved.dis = 0.0
    }

    if (msg.aoa_angle_horizontal !== undefined) {
      resolved.aoa_angle_horizontal = msg.aoa_angle_horizontal;
    }
    else {
      resolved.aoa_angle_horizontal = 0.0
    }

    if (msg.aoa_angle_vertical !== undefined) {
      resolved.aoa_angle_vertical = msg.aoa_angle_vertical;
    }
    else {
      resolved.aoa_angle_vertical = 0.0
    }

    if (msg.fp_rssi !== undefined) {
      resolved.fp_rssi = msg.fp_rssi;
    }
    else {
      resolved.fp_rssi = 0.0
    }

    if (msg.rx_rssi !== undefined) {
      resolved.rx_rssi = msg.rx_rssi;
    }
    else {
      resolved.rx_rssi = 0.0
    }

    if (msg.user_data !== undefined) {
      resolved.user_data = msg.user_data;
    }
    else {
      resolved.user_data = ''
    }

    return resolved;
    }
};

module.exports = IotFrame0Node;
