// Auto-generated. Do not edit!

// (in-package nlink_parser.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let TofsenseMFrame0Pixel = require('./TofsenseMFrame0Pixel.js');

//-----------------------------------------------------------

class TofsenseMFrame0 {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.id = null;
      this.system_time = null;
      this.pixel_count = null;
      this.pixels = null;
    }
    else {
      if (initObj.hasOwnProperty('id')) {
        this.id = initObj.id
      }
      else {
        this.id = 0;
      }
      if (initObj.hasOwnProperty('system_time')) {
        this.system_time = initObj.system_time
      }
      else {
        this.system_time = 0;
      }
      if (initObj.hasOwnProperty('pixel_count')) {
        this.pixel_count = initObj.pixel_count
      }
      else {
        this.pixel_count = 0;
      }
      if (initObj.hasOwnProperty('pixels')) {
        this.pixels = initObj.pixels
      }
      else {
        this.pixels = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TofsenseMFrame0
    // Serialize message field [id]
    bufferOffset = _serializer.uint8(obj.id, buffer, bufferOffset);
    // Serialize message field [system_time]
    bufferOffset = _serializer.uint32(obj.system_time, buffer, bufferOffset);
    // Serialize message field [pixel_count]
    bufferOffset = _serializer.uint8(obj.pixel_count, buffer, bufferOffset);
    // Serialize message field [pixels]
    // Serialize the length for message field [pixels]
    bufferOffset = _serializer.uint32(obj.pixels.length, buffer, bufferOffset);
    obj.pixels.forEach((val) => {
      bufferOffset = TofsenseMFrame0Pixel.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TofsenseMFrame0
    let len;
    let data = new TofsenseMFrame0(null);
    // Deserialize message field [id]
    data.id = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [system_time]
    data.system_time = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [pixel_count]
    data.pixel_count = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [pixels]
    // Deserialize array length for message field [pixels]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.pixels = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.pixels[i] = TofsenseMFrame0Pixel.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 7 * object.pixels.length;
    return length + 10;
  }

  static datatype() {
    // Returns string type for a message object
    return 'nlink_parser/TofsenseMFrame0';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '8b49d724777a20439f181e513a9f9e83';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 id
    uint32 system_time
    uint8 pixel_count
    TofsenseMFrame0Pixel[] pixels
    
    ================================================================================
    MSG: nlink_parser/TofsenseMFrame0Pixel
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
    const resolved = new TofsenseMFrame0(null);
    if (msg.id !== undefined) {
      resolved.id = msg.id;
    }
    else {
      resolved.id = 0
    }

    if (msg.system_time !== undefined) {
      resolved.system_time = msg.system_time;
    }
    else {
      resolved.system_time = 0
    }

    if (msg.pixel_count !== undefined) {
      resolved.pixel_count = msg.pixel_count;
    }
    else {
      resolved.pixel_count = 0
    }

    if (msg.pixels !== undefined) {
      resolved.pixels = new Array(msg.pixels.length);
      for (let i = 0; i < resolved.pixels.length; ++i) {
        resolved.pixels[i] = TofsenseMFrame0Pixel.Resolve(msg.pixels[i]);
      }
    }
    else {
      resolved.pixels = []
    }

    return resolved;
    }
};

module.exports = TofsenseMFrame0;
