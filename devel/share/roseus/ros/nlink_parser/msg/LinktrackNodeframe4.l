;; Auto-generated. Do not edit!


(when (boundp 'nlink_parser::LinktrackNodeframe4)
  (if (not (find-package "NLINK_PARSER"))
    (make-package "NLINK_PARSER"))
  (shadow 'LinktrackNodeframe4 (find-package "NLINK_PARSER")))
(unless (find-package "NLINK_PARSER::LINKTRACKNODEFRAME4")
  (make-package "NLINK_PARSER::LINKTRACKNODEFRAME4"))

(in-package "ROS")
;;//! \htmlinclude LinktrackNodeframe4.msg.html


(defclass nlink_parser::LinktrackNodeframe4
  :super ros::object
  :slots (_role _id _local_time _system_time _voltage _tags ))

(defmethod nlink_parser::LinktrackNodeframe4
  (:init
   (&key
    ((:role __role) 0)
    ((:id __id) 0)
    ((:local_time __local_time) 0)
    ((:system_time __system_time) 0)
    ((:voltage __voltage) 0.0)
    ((:tags __tags) ())
    )
   (send-super :init)
   (setq _role (round __role))
   (setq _id (round __id))
   (setq _local_time (round __local_time))
   (setq _system_time (round __system_time))
   (setq _voltage (float __voltage))
   (setq _tags __tags)
   self)
  (:role
   (&optional __role)
   (if __role (setq _role __role)) _role)
  (:id
   (&optional __id)
   (if __id (setq _id __id)) _id)
  (:local_time
   (&optional __local_time)
   (if __local_time (setq _local_time __local_time)) _local_time)
  (:system_time
   (&optional __system_time)
   (if __system_time (setq _system_time __system_time)) _system_time)
  (:voltage
   (&optional __voltage)
   (if __voltage (setq _voltage __voltage)) _voltage)
  (:tags
   (&rest __tags)
   (if (keywordp (car __tags))
       (send* _tags __tags)
     (progn
       (if __tags (setq _tags (car __tags)))
       _tags)))
  (:serialization-length
   ()
   (+
    ;; uint8 _role
    1
    ;; uint8 _id
    1
    ;; uint32 _local_time
    4
    ;; uint32 _system_time
    4
    ;; float32 _voltage
    4
    ;; nlink_parser/LinktrackNode4Tag[] _tags
    (apply #'+ (send-all _tags :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _role
       (write-byte _role s)
     ;; uint8 _id
       (write-byte _id s)
     ;; uint32 _local_time
       (write-long _local_time s)
     ;; uint32 _system_time
       (write-long _system_time s)
     ;; float32 _voltage
       (sys::poke _voltage (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; nlink_parser/LinktrackNode4Tag[] _tags
     (write-long (length _tags) s)
     (dolist (elem _tags)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _role
     (setq _role (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _id
     (setq _id (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint32 _local_time
     (setq _local_time (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _system_time
     (setq _system_time (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32 _voltage
     (setq _voltage (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; nlink_parser/LinktrackNode4Tag[] _tags
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _tags (let (r) (dotimes (i n) (push (instance nlink_parser::LinktrackNode4Tag :init) r)) r))
     (dolist (elem- _tags)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get nlink_parser::LinktrackNodeframe4 :md5sum-) "f3127a55ee0f30a49a0427a969f1b3ee")
(setf (get nlink_parser::LinktrackNodeframe4 :datatype-) "nlink_parser/LinktrackNodeframe4")
(setf (get nlink_parser::LinktrackNodeframe4 :definition-)
      "uint8 role
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

")



(provide :nlink_parser/LinktrackNodeframe4 "f3127a55ee0f30a49a0427a969f1b3ee")


