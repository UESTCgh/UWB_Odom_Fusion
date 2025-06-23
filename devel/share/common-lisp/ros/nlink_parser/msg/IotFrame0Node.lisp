; Auto-generated. Do not edit!


(cl:in-package nlink_parser-msg)


;//! \htmlinclude IotFrame0Node.msg.html

(cl:defclass <IotFrame0Node> (roslisp-msg-protocol:ros-message)
  ((uid
    :reader uid
    :initarg :uid
    :type cl:integer
    :initform 0)
   (dis
    :reader dis
    :initarg :dis
    :type cl:float
    :initform 0.0)
   (aoa_angle_horizontal
    :reader aoa_angle_horizontal
    :initarg :aoa_angle_horizontal
    :type cl:float
    :initform 0.0)
   (aoa_angle_vertical
    :reader aoa_angle_vertical
    :initarg :aoa_angle_vertical
    :type cl:float
    :initform 0.0)
   (fp_rssi
    :reader fp_rssi
    :initarg :fp_rssi
    :type cl:float
    :initform 0.0)
   (rx_rssi
    :reader rx_rssi
    :initarg :rx_rssi
    :type cl:float
    :initform 0.0)
   (user_data
    :reader user_data
    :initarg :user_data
    :type cl:string
    :initform ""))
)

(cl:defclass IotFrame0Node (<IotFrame0Node>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <IotFrame0Node>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'IotFrame0Node)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name nlink_parser-msg:<IotFrame0Node> is deprecated: use nlink_parser-msg:IotFrame0Node instead.")))

(cl:ensure-generic-function 'uid-val :lambda-list '(m))
(cl:defmethod uid-val ((m <IotFrame0Node>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:uid-val is deprecated.  Use nlink_parser-msg:uid instead.")
  (uid m))

(cl:ensure-generic-function 'dis-val :lambda-list '(m))
(cl:defmethod dis-val ((m <IotFrame0Node>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:dis-val is deprecated.  Use nlink_parser-msg:dis instead.")
  (dis m))

(cl:ensure-generic-function 'aoa_angle_horizontal-val :lambda-list '(m))
(cl:defmethod aoa_angle_horizontal-val ((m <IotFrame0Node>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:aoa_angle_horizontal-val is deprecated.  Use nlink_parser-msg:aoa_angle_horizontal instead.")
  (aoa_angle_horizontal m))

(cl:ensure-generic-function 'aoa_angle_vertical-val :lambda-list '(m))
(cl:defmethod aoa_angle_vertical-val ((m <IotFrame0Node>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:aoa_angle_vertical-val is deprecated.  Use nlink_parser-msg:aoa_angle_vertical instead.")
  (aoa_angle_vertical m))

(cl:ensure-generic-function 'fp_rssi-val :lambda-list '(m))
(cl:defmethod fp_rssi-val ((m <IotFrame0Node>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:fp_rssi-val is deprecated.  Use nlink_parser-msg:fp_rssi instead.")
  (fp_rssi m))

(cl:ensure-generic-function 'rx_rssi-val :lambda-list '(m))
(cl:defmethod rx_rssi-val ((m <IotFrame0Node>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:rx_rssi-val is deprecated.  Use nlink_parser-msg:rx_rssi instead.")
  (rx_rssi m))

(cl:ensure-generic-function 'user_data-val :lambda-list '(m))
(cl:defmethod user_data-val ((m <IotFrame0Node>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:user_data-val is deprecated.  Use nlink_parser-msg:user_data instead.")
  (user_data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <IotFrame0Node>) ostream)
  "Serializes a message object of type '<IotFrame0Node>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'uid)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'uid)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'uid)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'uid)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'dis))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'aoa_angle_horizontal))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'aoa_angle_vertical))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'fp_rssi))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rx_rssi))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'user_data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'user_data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <IotFrame0Node>) istream)
  "Deserializes a message object of type '<IotFrame0Node>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'uid)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'uid)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'uid)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'uid)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'dis) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'aoa_angle_horizontal) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'aoa_angle_vertical) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'fp_rssi) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rx_rssi) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'user_data) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'user_data) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<IotFrame0Node>)))
  "Returns string type for a message object of type '<IotFrame0Node>"
  "nlink_parser/IotFrame0Node")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'IotFrame0Node)))
  "Returns string type for a message object of type 'IotFrame0Node"
  "nlink_parser/IotFrame0Node")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<IotFrame0Node>)))
  "Returns md5sum for a message object of type '<IotFrame0Node>"
  "df423b5fa843bedeae14cfe493476452")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'IotFrame0Node)))
  "Returns md5sum for a message object of type 'IotFrame0Node"
  "df423b5fa843bedeae14cfe493476452")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<IotFrame0Node>)))
  "Returns full string definition for message of type '<IotFrame0Node>"
  (cl:format cl:nil "uint32 uid~%float32 dis~%float32 aoa_angle_horizontal~%float32 aoa_angle_vertical~%float32 fp_rssi~%float32 rx_rssi~%string user_data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'IotFrame0Node)))
  "Returns full string definition for message of type 'IotFrame0Node"
  (cl:format cl:nil "uint32 uid~%float32 dis~%float32 aoa_angle_horizontal~%float32 aoa_angle_vertical~%float32 fp_rssi~%float32 rx_rssi~%string user_data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <IotFrame0Node>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
     4 (cl:length (cl:slot-value msg 'user_data))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <IotFrame0Node>))
  "Converts a ROS message object to a list"
  (cl:list 'IotFrame0Node
    (cl:cons ':uid (uid msg))
    (cl:cons ':dis (dis msg))
    (cl:cons ':aoa_angle_horizontal (aoa_angle_horizontal msg))
    (cl:cons ':aoa_angle_vertical (aoa_angle_vertical msg))
    (cl:cons ':fp_rssi (fp_rssi msg))
    (cl:cons ':rx_rssi (rx_rssi msg))
    (cl:cons ':user_data (user_data msg))
))
