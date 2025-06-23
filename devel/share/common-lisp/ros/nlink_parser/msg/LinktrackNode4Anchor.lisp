; Auto-generated. Do not edit!


(cl:in-package nlink_parser-msg)


;//! \htmlinclude LinktrackNode4Anchor.msg.html

(cl:defclass <LinktrackNode4Anchor> (roslisp-msg-protocol:ros-message)
  ((id
    :reader id
    :initarg :id
    :type cl:fixnum
    :initform 0)
   (dis
    :reader dis
    :initarg :dis
    :type cl:float
    :initform 0.0))
)

(cl:defclass LinktrackNode4Anchor (<LinktrackNode4Anchor>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <LinktrackNode4Anchor>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'LinktrackNode4Anchor)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name nlink_parser-msg:<LinktrackNode4Anchor> is deprecated: use nlink_parser-msg:LinktrackNode4Anchor instead.")))

(cl:ensure-generic-function 'id-val :lambda-list '(m))
(cl:defmethod id-val ((m <LinktrackNode4Anchor>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:id-val is deprecated.  Use nlink_parser-msg:id instead.")
  (id m))

(cl:ensure-generic-function 'dis-val :lambda-list '(m))
(cl:defmethod dis-val ((m <LinktrackNode4Anchor>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader nlink_parser-msg:dis-val is deprecated.  Use nlink_parser-msg:dis instead.")
  (dis m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <LinktrackNode4Anchor>) ostream)
  "Serializes a message object of type '<LinktrackNode4Anchor>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'id)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'dis))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <LinktrackNode4Anchor>) istream)
  "Deserializes a message object of type '<LinktrackNode4Anchor>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'id)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'dis) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<LinktrackNode4Anchor>)))
  "Returns string type for a message object of type '<LinktrackNode4Anchor>"
  "nlink_parser/LinktrackNode4Anchor")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LinktrackNode4Anchor)))
  "Returns string type for a message object of type 'LinktrackNode4Anchor"
  "nlink_parser/LinktrackNode4Anchor")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<LinktrackNode4Anchor>)))
  "Returns md5sum for a message object of type '<LinktrackNode4Anchor>"
  "3449514afa3f54cb0e542c3e3a394e77")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'LinktrackNode4Anchor)))
  "Returns md5sum for a message object of type 'LinktrackNode4Anchor"
  "3449514afa3f54cb0e542c3e3a394e77")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<LinktrackNode4Anchor>)))
  "Returns full string definition for message of type '<LinktrackNode4Anchor>"
  (cl:format cl:nil "uint8 id~%float32 dis~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'LinktrackNode4Anchor)))
  "Returns full string definition for message of type 'LinktrackNode4Anchor"
  (cl:format cl:nil "uint8 id~%float32 dis~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <LinktrackNode4Anchor>))
  (cl:+ 0
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <LinktrackNode4Anchor>))
  "Converts a ROS message object to a list"
  (cl:list 'LinktrackNode4Anchor
    (cl:cons ':id (id msg))
    (cl:cons ':dis (dis msg))
))
