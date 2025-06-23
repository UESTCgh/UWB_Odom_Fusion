; Auto-generated. Do not edit!


(cl:in-package nlink_parser-msg)


;//! \htmlinclude LinktrackNode4Tag.msg.html

(cl:defclass <LinktrackNode4Tag> (roslisp-msg-protocol:ros-message)
  ((id
    :reader id
    :initarg :id
    :type cl:fixnum
    :initform 0)
   (voltage
    :reader voltage
    :initarg :voltage
    :type cl:float
    :initform 0.0)
   (anchors
    :reader anchors
    :initarg :anchors
    :type (cl:vector nlink_parser-msg:LinktrackNode4Anchor)
   :initform (cl:make-array 0 :element-type 'nlink_parser-msg:LinktrackNode4Anchor :initial-element (cl:make-instance 'nlink_parser-msg:LinktrackNode4Anchor))))
)

(cl:defclass LinktrackNode4Tag (<LinktrackNode4Tag>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <LinktrackNode4Tag>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'LinktrackNode4Tag)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name nlink_parser-msg:<LinktrackNode4Tag> is deprecated: use nlink_parser-msg:LinktrackNode4Tag instead.")))

(cl:ensure-generic-function 'id-val :lambda-list '(m))
(cl:defmethod id-val ((m <LinktrackNode4Tag>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:id-val is deprecated.  Use nlink_parser-msg:id instead.")
  (id m))

(cl:ensure-generic-function 'voltage-val :lambda-list '(m))
(cl:defmethod voltage-val ((m <LinktrackNode4Tag>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:voltage-val is deprecated.  Use nlink_parser-msg:voltage instead.")
  (voltage m))

(cl:ensure-generic-function 'anchors-val :lambda-list '(m))
(cl:defmethod anchors-val ((m <LinktrackNode4Tag>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:anchors-val is deprecated.  Use nlink_parser-msg:anchors instead.")
  (anchors m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <LinktrackNode4Tag>) ostream)
  "Serializes a message object of type '<LinktrackNode4Tag>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'id)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'voltage))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'anchors))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'anchors))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <LinktrackNode4Tag>) istream)
  "Deserializes a message object of type '<LinktrackNode4Tag>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'id)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'voltage) (roslisp-utils:decode-single-float-bits bits)))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'anchors) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'anchors)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'nlink_parser-msg:LinktrackNode4Anchor))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<LinktrackNode4Tag>)))
  "Returns string type for a message object of type '<LinktrackNode4Tag>"
  "nlink_parser/LinktrackNode4Tag")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LinktrackNode4Tag)))
  "Returns string type for a message object of type 'LinktrackNode4Tag"
  "nlink_parser/LinktrackNode4Tag")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<LinktrackNode4Tag>)))
  "Returns md5sum for a message object of type '<LinktrackNode4Tag>"
  "52d7d856087ab9caa357ff4bd2e80950")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'LinktrackNode4Tag)))
  "Returns md5sum for a message object of type 'LinktrackNode4Tag"
  "52d7d856087ab9caa357ff4bd2e80950")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<LinktrackNode4Tag>)))
  "Returns full string definition for message of type '<LinktrackNode4Tag>"
  (cl:format cl:nil "uint8 id~%float32 voltage~%LinktrackNode4Anchor[] anchors~%~%================================================================================~%MSG: nlink_parser/LinktrackNode4Anchor~%uint8 id~%float32 dis~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'LinktrackNode4Tag)))
  "Returns full string definition for message of type 'LinktrackNode4Tag"
  (cl:format cl:nil "uint8 id~%float32 voltage~%LinktrackNode4Anchor[] anchors~%~%================================================================================~%MSG: nlink_parser/LinktrackNode4Anchor~%uint8 id~%float32 dis~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <LinktrackNode4Tag>))
  (cl:+ 0
     1
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'anchors) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <LinktrackNode4Tag>))
  "Converts a ROS message object to a list"
  (cl:list 'LinktrackNode4Tag
    (cl:cons ':id (id msg))
    (cl:cons ':voltage (voltage msg))
    (cl:cons ':anchors (anchors msg))
))
