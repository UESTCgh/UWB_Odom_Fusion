; Auto-generated. Do not edit!


(cl:in-package nlink_parser-msg)


;//! \htmlinclude TofsenseMFrame0.msg.html

(cl:defclass <TofsenseMFrame0> (roslisp-msg-protocol:ros-message)
  ((id
    :reader id
    :initarg :id
    :type cl:fixnum
    :initform 0)
   (system_time
    :reader system_time
    :initarg :system_time
    :type cl:integer
    :initform 0)
   (pixel_count
    :reader pixel_count
    :initarg :pixel_count
    :type cl:fixnum
    :initform 0)
   (pixels
    :reader pixels
    :initarg :pixels
    :type (cl:vector nlink_parser-msg:TofsenseMFrame0Pixel)
   :initform (cl:make-array 0 :element-type 'nlink_parser-msg:TofsenseMFrame0Pixel :initial-element (cl:make-instance 'nlink_parser-msg:TofsenseMFrame0Pixel))))
)

(cl:defclass TofsenseMFrame0 (<TofsenseMFrame0>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TofsenseMFrame0>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TofsenseMFrame0)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name nlink_parser-msg:<TofsenseMFrame0> is deprecated: use nlink_parser-msg:TofsenseMFrame0 instead.")))

(cl:ensure-generic-function 'id-val :lambda-list '(m))
(cl:defmethod id-val ((m <TofsenseMFrame0>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:id-val is deprecated.  Use nlink_parser-msg:id instead.")
  (id m))

(cl:ensure-generic-function 'system_time-val :lambda-list '(m))
(cl:defmethod system_time-val ((m <TofsenseMFrame0>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:system_time-val is deprecated.  Use nlink_parser-msg:system_time instead.")
  (system_time m))

(cl:ensure-generic-function 'pixel_count-val :lambda-list '(m))
(cl:defmethod pixel_count-val ((m <TofsenseMFrame0>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:pixel_count-val is deprecated.  Use nlink_parser-msg:pixel_count instead.")
  (pixel_count m))

(cl:ensure-generic-function 'pixels-val :lambda-list '(m))
(cl:defmethod pixels-val ((m <TofsenseMFrame0>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:pixels-val is deprecated.  Use nlink_parser-msg:pixels instead.")
  (pixels m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TofsenseMFrame0>) ostream)
  "Serializes a message object of type '<TofsenseMFrame0>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'id)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'system_time)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'system_time)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'system_time)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'system_time)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'pixel_count)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'pixels))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'pixels))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TofsenseMFrame0>) istream)
  "Deserializes a message object of type '<TofsenseMFrame0>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'id)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'system_time)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'system_time)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'system_time)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'system_time)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'pixel_count)) (cl:read-byte istream))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'pixels) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'pixels)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'nlink_parser-msg:TofsenseMFrame0Pixel))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TofsenseMFrame0>)))
  "Returns string type for a message object of type '<TofsenseMFrame0>"
  "nlink_parser/TofsenseMFrame0")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TofsenseMFrame0)))
  "Returns string type for a message object of type 'TofsenseMFrame0"
  "nlink_parser/TofsenseMFrame0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TofsenseMFrame0>)))
  "Returns md5sum for a message object of type '<TofsenseMFrame0>"
  "8b49d724777a20439f181e513a9f9e83")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TofsenseMFrame0)))
  "Returns md5sum for a message object of type 'TofsenseMFrame0"
  "8b49d724777a20439f181e513a9f9e83")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TofsenseMFrame0>)))
  "Returns full string definition for message of type '<TofsenseMFrame0>"
  (cl:format cl:nil "uint8 id~%uint32 system_time~%uint8 pixel_count~%TofsenseMFrame0Pixel[] pixels~%~%================================================================================~%MSG: nlink_parser/TofsenseMFrame0Pixel~%float32 dis~%uint8 dis_status~%uint16 signal_strength~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TofsenseMFrame0)))
  "Returns full string definition for message of type 'TofsenseMFrame0"
  (cl:format cl:nil "uint8 id~%uint32 system_time~%uint8 pixel_count~%TofsenseMFrame0Pixel[] pixels~%~%================================================================================~%MSG: nlink_parser/TofsenseMFrame0Pixel~%float32 dis~%uint8 dis_status~%uint16 signal_strength~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TofsenseMFrame0>))
  (cl:+ 0
     1
     4
     1
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'pixels) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TofsenseMFrame0>))
  "Converts a ROS message object to a list"
  (cl:list 'TofsenseMFrame0
    (cl:cons ':id (id msg))
    (cl:cons ':system_time (system_time msg))
    (cl:cons ':pixel_count (pixel_count msg))
    (cl:cons ':pixels (pixels msg))
))
