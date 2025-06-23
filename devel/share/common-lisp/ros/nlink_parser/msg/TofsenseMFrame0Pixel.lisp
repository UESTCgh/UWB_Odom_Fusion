; Auto-generated. Do not edit!


(cl:in-package nlink_parser-msg)


;//! \htmlinclude TofsenseMFrame0Pixel.msg.html

(cl:defclass <TofsenseMFrame0Pixel> (roslisp-msg-protocol:ros-message)
  ((dis
    :reader dis
    :initarg :dis
    :type cl:float
    :initform 0.0)
   (dis_status
    :reader dis_status
    :initarg :dis_status
    :type cl:fixnum
    :initform 0)
   (signal_strength
    :reader signal_strength
    :initarg :signal_strength
    :type cl:fixnum
    :initform 0))
)

(cl:defclass TofsenseMFrame0Pixel (<TofsenseMFrame0Pixel>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TofsenseMFrame0Pixel>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TofsenseMFrame0Pixel)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name nlink_parser-msg:<TofsenseMFrame0Pixel> is deprecated: use nlink_parser-msg:TofsenseMFrame0Pixel instead.")))

(cl:ensure-generic-function 'dis-val :lambda-list '(m))
(cl:defmethod dis-val ((m <TofsenseMFrame0Pixel>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:dis-val is deprecated.  Use nlink_parser-msg:dis instead.")
  (dis m))

(cl:ensure-generic-function 'dis_status-val :lambda-list '(m))
(cl:defmethod dis_status-val ((m <TofsenseMFrame0Pixel>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:dis_status-val is deprecated.  Use nlink_parser-msg:dis_status instead.")
  (dis_status m))

(cl:ensure-generic-function 'signal_strength-val :lambda-list '(m))
(cl:defmethod signal_strength-val ((m <TofsenseMFrame0Pixel>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:signal_strength-val is deprecated.  Use nlink_parser-msg:signal_strength instead.")
  (signal_strength m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TofsenseMFrame0Pixel>) ostream)
  "Serializes a message object of type '<TofsenseMFrame0Pixel>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'dis))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'dis_status)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'signal_strength)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'signal_strength)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TofsenseMFrame0Pixel>) istream)
  "Deserializes a message object of type '<TofsenseMFrame0Pixel>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'dis) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'dis_status)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'signal_strength)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'signal_strength)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TofsenseMFrame0Pixel>)))
  "Returns string type for a message object of type '<TofsenseMFrame0Pixel>"
  "nlink_parser/TofsenseMFrame0Pixel")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TofsenseMFrame0Pixel)))
  "Returns string type for a message object of type 'TofsenseMFrame0Pixel"
  "nlink_parser/TofsenseMFrame0Pixel")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TofsenseMFrame0Pixel>)))
  "Returns md5sum for a message object of type '<TofsenseMFrame0Pixel>"
  "d8d02fd7942fc6660671b1c9411abd9c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TofsenseMFrame0Pixel)))
  "Returns md5sum for a message object of type 'TofsenseMFrame0Pixel"
  "d8d02fd7942fc6660671b1c9411abd9c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TofsenseMFrame0Pixel>)))
  "Returns full string definition for message of type '<TofsenseMFrame0Pixel>"
  (cl:format cl:nil "float32 dis~%uint8 dis_status~%uint16 signal_strength~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TofsenseMFrame0Pixel)))
  "Returns full string definition for message of type 'TofsenseMFrame0Pixel"
  (cl:format cl:nil "float32 dis~%uint8 dis_status~%uint16 signal_strength~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TofsenseMFrame0Pixel>))
  (cl:+ 0
     4
     1
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TofsenseMFrame0Pixel>))
  "Converts a ROS message object to a list"
  (cl:list 'TofsenseMFrame0Pixel
    (cl:cons ':dis (dis msg))
    (cl:cons ':dis_status (dis_status msg))
    (cl:cons ':signal_strength (signal_strength msg))
))
