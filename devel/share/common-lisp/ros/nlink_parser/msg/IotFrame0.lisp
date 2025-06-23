; Auto-generated. Do not edit!


(cl:in-package nlink_parser-msg)


;//! \htmlinclude IotFrame0.msg.html

(cl:defclass <IotFrame0> (roslisp-msg-protocol:ros-message)
  ((uid
    :reader uid
    :initarg :uid
    :type cl:integer
    :initform 0)
   (system_time
    :reader system_time
    :initarg :system_time
    :type cl:integer
    :initform 0)
   (io_status
    :reader io_status
    :initarg :io_status
    :type cl:fixnum
    :initform 0)
   (nodes
    :reader nodes
    :initarg :nodes
    :type (cl:vector nlink_parser-msg:IotFrame0Node)
   :initform (cl:make-array 0 :element-type 'nlink_parser-msg:IotFrame0Node :initial-element (cl:make-instance 'nlink_parser-msg:IotFrame0Node))))
)

(cl:defclass IotFrame0 (<IotFrame0>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <IotFrame0>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'IotFrame0)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name nlink_parser-msg:<IotFrame0> is deprecated: use nlink_parser-msg:IotFrame0 instead.")))

(cl:ensure-generic-function 'uid-val :lambda-list '(m))
(cl:defmethod uid-val ((m <IotFrame0>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:uid-val is deprecated.  Use nlink_parser-msg:uid instead.")
  (uid m))

(cl:ensure-generic-function 'system_time-val :lambda-list '(m))
(cl:defmethod system_time-val ((m <IotFrame0>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:system_time-val is deprecated.  Use nlink_parser-msg:system_time instead.")
  (system_time m))

(cl:ensure-generic-function 'io_status-val :lambda-list '(m))
(cl:defmethod io_status-val ((m <IotFrame0>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:io_status-val is deprecated.  Use nlink_parser-msg:io_status instead.")
  (io_status m))

(cl:ensure-generic-function 'nodes-val :lambda-list '(m))
(cl:defmethod nodes-val ((m <IotFrame0>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:nodes-val is deprecated.  Use nlink_parser-msg:nodes instead.")
  (nodes m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <IotFrame0>) ostream)
  "Serializes a message object of type '<IotFrame0>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'uid)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'uid)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'uid)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'uid)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'system_time)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'system_time)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'system_time)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'system_time)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'io_status)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'nodes))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'nodes))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <IotFrame0>) istream)
  "Deserializes a message object of type '<IotFrame0>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'uid)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'uid)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'uid)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'uid)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'system_time)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'system_time)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'system_time)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'system_time)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'io_status)) (cl:read-byte istream))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'nodes) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'nodes)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'nlink_parser-msg:IotFrame0Node))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<IotFrame0>)))
  "Returns string type for a message object of type '<IotFrame0>"
  "nlink_parser/IotFrame0")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'IotFrame0)))
  "Returns string type for a message object of type 'IotFrame0"
  "nlink_parser/IotFrame0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<IotFrame0>)))
  "Returns md5sum for a message object of type '<IotFrame0>"
  "8854697b4ab999e932111a65ada95ab0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'IotFrame0)))
  "Returns md5sum for a message object of type 'IotFrame0"
  "8854697b4ab999e932111a65ada95ab0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<IotFrame0>)))
  "Returns full string definition for message of type '<IotFrame0>"
  (cl:format cl:nil "uint32 uid~%uint32 system_time~%uint8 io_status~%IotFrame0Node[] nodes~%~%================================================================================~%MSG: nlink_parser/IotFrame0Node~%uint32 uid~%float32 dis~%float32 aoa_angle_horizontal~%float32 aoa_angle_vertical~%float32 fp_rssi~%float32 rx_rssi~%string user_data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'IotFrame0)))
  "Returns full string definition for message of type 'IotFrame0"
  (cl:format cl:nil "uint32 uid~%uint32 system_time~%uint8 io_status~%IotFrame0Node[] nodes~%~%================================================================================~%MSG: nlink_parser/IotFrame0Node~%uint32 uid~%float32 dis~%float32 aoa_angle_horizontal~%float32 aoa_angle_vertical~%float32 fp_rssi~%float32 rx_rssi~%string user_data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <IotFrame0>))
  (cl:+ 0
     4
     4
     1
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'nodes) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <IotFrame0>))
  "Converts a ROS message object to a list"
  (cl:list 'IotFrame0
    (cl:cons ':uid (uid msg))
    (cl:cons ':system_time (system_time msg))
    (cl:cons ':io_status (io_status msg))
    (cl:cons ':nodes (nodes msg))
))
