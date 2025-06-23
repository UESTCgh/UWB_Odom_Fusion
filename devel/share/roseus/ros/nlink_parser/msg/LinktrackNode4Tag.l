;; Auto-generated. Do not edit!


(when (boundp 'nlink_parser::LinktrackNode4Tag)
  (if (not (find-package "NLINK_PARSER"))
    (make-package "NLINK_PARSER"))
  (shadow 'LinktrackNode4Tag (find-package "NLINK_PARSER")))
(unless (find-package "NLINK_PARSER::LINKTRACKNODE4TAG")
  (make-package "NLINK_PARSER::LINKTRACKNODE4TAG"))

(in-package "ROS")
;;//! \htmlinclude LinktrackNode4Tag.msg.html


(defclass nlink_parser::LinktrackNode4Tag
  :super ros::object
  :slots (_id _voltage _anchors ))

(defmethod nlink_parser::LinktrackNode4Tag
  (:init
   (&key
    ((:id __id) 0)
    ((:voltage __voltage) 0.0)
    ((:anchors __anchors) ())
    )
   (send-super :init)
   (setq _id (round __id))
   (setq _voltage (float __voltage))
   (setq _anchors __anchors)
   self)
  (:id
   (&optional __id)
   (if __id (setq _id __id)) _id)
  (:voltage
   (&optional __voltage)
   (if __voltage (setq _voltage __voltage)) _voltage)
  (:anchors
   (&rest __anchors)
   (if (keywordp (car __anchors))
       (send* _anchors __anchors)
     (progn
       (if __anchors (setq _anchors (car __anchors)))
       _anchors)))
  (:serialization-length
   ()
   (+
    ;; uint8 _id
    1
    ;; float32 _voltage
    4
    ;; nlink_parser/LinktrackNode4Anchor[] _anchors
    (apply #'+ (send-all _anchors :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _id
       (write-byte _id s)
     ;; float32 _voltage
       (sys::poke _voltage (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; nlink_parser/LinktrackNode4Anchor[] _anchors
     (write-long (length _anchors) s)
     (dolist (elem _anchors)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _id
     (setq _id (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float32 _voltage
     (setq _voltage (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; nlink_parser/LinktrackNode4Anchor[] _anchors
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _anchors (let (r) (dotimes (i n) (push (instance nlink_parser::LinktrackNode4Anchor :init) r)) r))
     (dolist (elem- _anchors)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get nlink_parser::LinktrackNode4Tag :md5sum-) "52d7d856087ab9caa357ff4bd2e80950")
(setf (get nlink_parser::LinktrackNode4Tag :datatype-) "nlink_parser/LinktrackNode4Tag")
(setf (get nlink_parser::LinktrackNode4Tag :definition-)
      "uint8 id
float32 voltage
LinktrackNode4Anchor[] anchors

================================================================================
MSG: nlink_parser/LinktrackNode4Anchor
uint8 id
float32 dis

")



(provide :nlink_parser/LinktrackNode4Tag "52d7d856087ab9caa357ff4bd2e80950")


