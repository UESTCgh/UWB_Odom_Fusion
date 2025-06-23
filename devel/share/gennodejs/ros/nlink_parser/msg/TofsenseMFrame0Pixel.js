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

class TofsenseMFrame0Pixel {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.dis = null;
      this.dis_status = null;
      this.signal_strength = null;
    }
    else {
      if (initObj.hasOwnProperty('dis')) {
        this.dis = initObj.dis
      }
      else {
        this.dis = 0.0;
      }
      if (initObj.hasOwnProperty('dis_status')) {
        this.dis_status = initObj.dis_status
      }
      else {
        this.dis_status = 0;
      }
      if (initObj.hasOwnProperty('signal_strength')) {
        this.signal_strength = initObj.signal_strength
      }
      else {
        this.signal_strength = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TofsenseMFrame0Pixel
    // Serialize message field [dis]
    bufferOffset = _serializer.float32(obj.dis, buffer, bufferOffset);
    // Serialize message field [dis_status]
    bufferOffset = _serializer.uint8(obj.dis_status, buffer, bufferOffset);
    // Serialize message field [signal_strength]
    bufferOffset = _serializer.uint16(obj.signal_strength, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TofsenseMFrame0Pixel
    let len;
    let data = new TofsenseMFrame0Pixel(null);
    // Deserialize message field [dis]
    data.dis = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [dis_status]
    data.dis_status = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [signal_strength]
    data.signal_strength = _deserializer.uint16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 7;
  }

  static datatype() {
    // Returns string type for a message object
    return 'nlink_parser/TofsenseMFrame0Pixel';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd8d02fd7942fc6660671b1c9411abd9c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 dis
    uint8 dis_status
    uint16 signal_strength
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TofsenseMFrame0Pixel(null);
    if (msg.dis !== undefined) {
      resolved.dis = msg.dis;
    }
    else {
      resolved.dis = 0.0
    }

    if (msg.dis_status !== undefined) {
      resolved.dis_status = msg.dis_status;
    }
    else {
      resolved.dis_status = 0
    }

    if (msg.signal_strength !== undefined) {
      resolved.signal_strength = msg.signal_strength;
    }
    else {
      resolved.signal_strength = 0
    }

    return resolved;
    }
};

module.exports = TofsenseMFrame0Pixel;
