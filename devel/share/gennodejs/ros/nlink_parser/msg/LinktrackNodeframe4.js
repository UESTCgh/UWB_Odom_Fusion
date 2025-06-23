// Auto-generated. Do not edit!

// (in-package nlink_parser.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let LinktrackNode4Tag = require('./LinktrackNode4Tag.js');

//-----------------------------------------------------------

class LinktrackNodeframe4 {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.role = null;
      this.id = null;
      this.local_time = null;
      this.system_time = null;
      this.voltage = null;
      this.tags = null;
    }
    else {
      if (initObj.hasOwnProperty('role')) {
        this.role = initObj.role
      }
      else {
        this.role = 0;
      }
      if (initObj.hasOwnProperty('id')) {
        this.id = initObj.id
      }
      else {
        this.id = 0;
      }
      if (initObj.hasOwnProperty('local_time')) {
        this.local_time = initObj.local_time
      }
      else {
        this.local_time = 0;
      }
      if (initObj.hasOwnProperty('system_time')) {
        this.system_time = initObj.system_time
      }
      else {
        this.system_time = 0;
      }
      if (initObj.hasOwnProperty('voltage')) {
        this.voltage = initObj.voltage
      }
      else {
        this.voltage = 0.0;
      }
      if (initObj.hasOwnProperty('tags')) {
        this.tags = initObj.tags
      }
      else {
        this.tags = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type LinktrackNodeframe4
    // Serialize message field [role]
    bufferOffset = _serializer.uint8(obj.role, buffer, bufferOffset);
    // Serialize message field [id]
    bufferOffset = _serializer.uint8(obj.id, buffer, bufferOffset);
    // Serialize message field [local_time]
    bufferOffset = _serializer.uint32(obj.local_time, buffer, bufferOffset);
    // Serialize message field [system_time]
    bufferOffset = _serializer.uint32(obj.system_time, buffer, bufferOffset);
    // Serialize message field [voltage]
    bufferOffset = _serializer.float32(obj.voltage, buffer, bufferOffset);
    // Serialize message field [tags]
    // Serialize the length for message field [tags]
    bufferOffset = _serializer.uint32(obj.tags.length, buffer, bufferOffset);
    obj.tags.forEach((val) => {
      bufferOffset = LinktrackNode4Tag.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type LinktrackNodeframe4
    let len;
    let data = new LinktrackNodeframe4(null);
    // Deserialize message field [role]
    data.role = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [id]
    data.id = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [local_time]
    data.local_time = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [system_time]
    data.system_time = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [voltage]
    data.voltage = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [tags]
    // Deserialize array length for message field [tags]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.tags = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.tags[i] = LinktrackNode4Tag.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.tags.forEach((val) => {
      length += LinktrackNode4Tag.getMessageSize(val);
    });
    return length + 18;
  }

  static datatype() {
    // Returns string type for a message object
    return 'nlink_parser/LinktrackNodeframe4';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'f3127a55ee0f30a49a0427a969f1b3ee';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 role
    uint8 id
    uint32 local_time
    uint32 system_time
    float32 voltage
    LinktrackNode4Tag[] tags
    
    ================================================================================
    MSG: nlink_parser/LinktrackNode4Tag
    uint8 id
    float32 voltage
    LinktrackNode4Anchor[] anchors
    
    ================================================================================
    MSG: nlink_parser/LinktrackNode4Anchor
    uint8 id
    float32 dis
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new LinktrackNodeframe4(null);
    if (msg.role !== undefined) {
      resolved.role = msg.role;
    }
    else {
      resolved.role = 0
    }

    if (msg.id !== undefined) {
      resolved.id = msg.id;
    }
    else {
      resolved.id = 0
    }

    if (msg.local_time !== undefined) {
      resolved.local_time = msg.local_time;
    }
    else {
      resolved.local_time = 0
    }

    if (msg.system_time !== undefined) {
      resolved.system_time = msg.system_time;
    }
    else {
      resolved.system_time = 0
    }

    if (msg.voltage !== undefined) {
      resolved.voltage = msg.voltage;
    }
    else {
      resolved.voltage = 0.0
    }

    if (msg.tags !== undefined) {
      resolved.tags = new Array(msg.tags.length);
      for (let i = 0; i < resolved.tags.length; ++i) {
        resolved.tags[i] = LinktrackNode4Tag.Resolve(msg.tags[i]);
      }
    }
    else {
      resolved.tags = []
    }

    return resolved;
    }
};

module.exports = LinktrackNodeframe4;
