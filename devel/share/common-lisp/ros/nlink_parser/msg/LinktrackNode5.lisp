; Auto-generated. Do not edit!


(cl:in-package nlink_parser-msg)


;//! \htmlinclude LinktrackNode5.msg.html

(cl:defclass <LinktrackNode5> (roslisp-msg-protocol:ros-message)
  ((role
    :reader role
    :initarg :role
    :type cl:fixnum
    :initform 0)
   (id
    :reader id
    :initarg :id
    :type cl:integer
    :initform 0)
   (dis
    :reader dis
    :initarg :dis
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
    :initform 0.0))
)

(cl:defclass LinktrackNode5 (<LinktrackNode5>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <LinktrackNode5>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'LinktrackNode5)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name nlink_parser-msg:<LinktrackNode5> is deprecated: use nlink_parser-msg:LinktrackNode5 instead.")))

(cl:ensure-generic-function 'role-val :lambda-list '(m))
(cl:defmethod role-val ((m <LinktrackNode5>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:role-val is deprecated.  Use nlink_parser-msg:role instead.")
  (role m))

(cl:ensure-generic-function 'id-val :lambda-list '(m))
(cl:defmethod id-val ((m <LinktrackNode5>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:id-val is deprecated.  Use nlink_parser-msg:id instead.")
  (id m))

(cl:ensure-generic-function 'dis-val :lambda-list '(m))
(cl:defmethod dis-val ((m <LinktrackNode5>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:dis-val is deprecated.  Use nlink_parser-msg:dis instead.")
  (dis m))

(cl:ensure-generic-function 'fp_rssi-val :lambda-list '(m))
(cl:defmethod fp_rssi-val ((m <LinktrackNode5>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:fp_rssi-val is deprecated.  Use nlink_parser-msg:fp_rssi instead.")
  (fp_rssi m))

(cl:ensure-generic-function 'rx_rssi-val :lambda-list '(m))
(cl:defmethod rx_rssi-val ((m <LinktrackNode5>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:rx_rssi-val is deprecated.  Use nlink_parser-msg:rx_rssi instead.")
  (rx_rssi m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <LinktrackNode5>) ostream)
  "Serializes a message object of type '<LinktrackNode5>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'role)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'id)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'dis))))
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
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <LinktrackNode5>) istream)
  "Deserializes a message object of type '<LinktrackNode5>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'role)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'id)) (cl:read-byte istream))
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
    (cl:setf (cl:slot-value msg 'fp_rssi) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rx_rssi) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<LinktrackNode5>)))
  "Returns string type for a message object of type '<LinktrackNode5>"
  "nlink_parser/LinktrackNode5")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LinktrackNode5)))
  "Returns string type for a message object of type 'LinktrackNode5"
  "nlink_parser/LinktrackNode5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<LinktrackNode5>)))
  "Returns md5sum for a message object of type '<LinktrackNode5>"
  "a9a6a698583281aaa5b8500c404c95b6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'LinktrackNode5)))
  "Returns md5sum for a message object of type 'LinktrackNode5"
  "a9a6a698583281aaa5b8500c404c95b6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<LinktrackNode5>)))
  "Returns full string definition for message of type '<LinktrackNode5>"
  (cl:format cl:nil "uint8 role~%uint32 id~%float32 dis~%float32 fp_rssi~%float32 rx_rssi~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'LinktrackNode5)))
  "Returns full string definition for message of type 'LinktrackNode5"
  (cl:format cl:nil "uint8 role~%uint32 id~%float32 dis~%float32 fp_rssi~%float32 rx_rssi~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <LinktrackNode5>))
  (cl:+ 0
     1
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <LinktrackNode5>))
  "Converts a ROS message object to a list"
  (cl:list 'LinktrackNode5
    (cl:cons ':role (role msg))
    (cl:cons ':id (id msg))
    (cl:cons ':dis (dis msg))
    (cl:cons ':fp_rssi (fp_rssi msg))
    (cl:cons ':rx_rssi (rx_rssi msg))
))
