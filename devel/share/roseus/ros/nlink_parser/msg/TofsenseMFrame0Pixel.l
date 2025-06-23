;; Auto-generated. Do not edit!


(when (boundp 'nlink_parser::TofsenseMFrame0Pixel)
  (if (not (find-package "NLINK_PARSER"))
    (make-package "NLINK_PARSER"))
  (shadow 'TofsenseMFrame0Pixel (find-package "NLINK_PARSER")))
(unless (find-package "NLINK_PARSER::TOFSENSEMFRAME0PIXEL")
  (make-package "NLINK_PARSER::TOFSENSEMFRAME0PIXEL"))

(in-package "ROS")
;;//! \htmlinclude TofsenseMFrame0Pixel.msg.html


(defclass nlink_parser::TofsenseMFrame0Pixel
  :super ros::object
  :slots (_dis _dis_status _signal_strength ))

(defmethod nlink_parser::TofsenseMFrame0Pixel
  (:init
   (&key
    ((:dis __dis) 0.0)
    ((:dis_status __dis_status) 0)
    ((:signal_strength __signal_strength) 0)
    )
   (send-super :init)
   (setq _dis (float __dis))
   (setq _dis_status (round __dis_status))
   (setq _signal_strength (round __signal_strength))
   self)
  (:dis
   (&optional __dis)
   (if __dis (setq _dis __dis)) _dis)
  (:dis_status
   (&optional __dis_status)
   (if __dis_status (setq _dis_status __dis_status)) _dis_status)
  (:signal_strength
   (&optional __signal_strength)
   (if __signal_strength (setq _signal_strength __signal_strength)) _signal_strength)
  (:serialization-length
   ()
   (+
    ;; float32 _dis
    4
    ;; uint8 _dis_status
    1
    ;; uint16 _signal_strength
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _dis
       (sys::poke _dis (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint8 _dis_status
       (write-byte _dis_status s)
     ;; uint16 _signal_strength
       (write-word _signal_strength s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _dis
     (setq _dis (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint8 _dis_status
     (setq _dis_status (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint16 _signal_strength
     (setq _signal_strength (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get nlink_parser::TofsenseMFrame0Pixel :md5sum-) "d8d02fd7942fc6660671b1c9411abd9c")
(setf (get nlink_parser::TofsenseMFrame0Pixel :datatype-) "nlink_parser/TofsenseMFrame0Pixel")
(setf (get nlink_parser::TofsenseMFrame0Pixel :definition-)
      "float32 dis
uint8 dis_status
uint16 signal_strength

")



(provide :nlink_parser/TofsenseMFrame0Pixel "d8d02fd7942fc6660671b1c9411abd9c")


