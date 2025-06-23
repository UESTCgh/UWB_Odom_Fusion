// Auto-generated. Do not edit!

// (in-package nlink_parser.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let LinktrackNode4Anchor = require('./LinktrackNode4Anchor.js');

//-----------------------------------------------------------

class LinktrackNode4Tag {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.id = null;
      this.voltage = null;
      this.anchors = null;
    }
    else {
      if (initObj.hasOwnProperty('id')) {
        this.id = initObj.id
      }
      else {
        this.id = 0;
      }
      if (initObj.hasOwnProperty('voltage')) {
        this.voltage = initObj.voltage
      }
      else {
        this.voltage = 0.0;
      }
      if (initObj.hasOwnProperty('anchors')) {
        this.anchors = initObj.anchors
      }
      else {
        this.anchors = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type LinktrackNode4Tag
    // Serialize message field [id]
    bufferOffset = _serializer.uint8(obj.id, buffer, bufferOffset);
    // Serialize message field [voltage]
    bufferOffset = _serializer.float32(obj.voltage, buffer, bufferOffset);
    // Serialize message field [anchors]
    // Serialize the length for message field [anchors]
    bufferOffset = _serializer.uint32(obj.anchors.length, buffer, bufferOffset);
    obj.anchors.forEach((val) => {
      bufferOffset = LinktrackNode4Anchor.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type LinktrackNode4Tag
    let len;
    let data = new LinktrackNode4Tag(null);
    // Deserialize message field [id]
    data.id = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [voltage]
    data.voltage = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [anchors]
    // Deserialize array length for message field [anchors]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.anchors = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.anchors[i] = LinktrackNode4Anchor.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 5 * object.anchors.length;
    return length + 9;
  }

  static datatype() {
    // Returns string type for a message object
    return 'nlink_parser/LinktrackNode4Tag';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '52d7d856087ab9caa357ff4bd2e80950';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    const resolved = new LinktrackNode4Tag(null);
    if (msg.id !== undefined) {
      resolved.id = msg.id;
    }
    else {
      resolved.id = 0
    }

    if (msg.voltage !== undefined) {
      resolved.voltage = msg.voltage;
    }
    else {
      resolved.voltage = 0.0
    }

    if (msg.anchors !== undefined) {
      resolved.anchors = new Array(msg.anchors.length);
      for (let i = 0; i < resolved.anchors.length; ++i) {
        resolved.anchors[i] = LinktrackNode4Anchor.Resolve(msg.anchors[i]);
      }
    }
    else {
      resolved.anchors = []
    }

    return resolved;
    }
};

module.exports = LinktrackNode4Tag;
