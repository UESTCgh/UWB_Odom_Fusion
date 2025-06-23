;; Auto-generated. Do not edit!


(when (boundp 'nlink_parser::TofsenseMFrame0)
  (if (not (find-package "NLINK_PARSER"))
    (make-package "NLINK_PARSER"))
  (shadow 'TofsenseMFrame0 (find-package "NLINK_PARSER")))
(unless (find-package "NLINK_PARSER::TOFSENSEMFRAME0")
  (make-package "NLINK_PARSER::TOFSENSEMFRAME0"))

(in-package "ROS")
;;//! \htmlinclude TofsenseMFrame0.msg.html


(defclass nlink_parser::TofsenseMFrame0
  :super ros::object
  :slots (_id _system_time _pixel_count _pixels ))

(defmethod nlink_parser::TofsenseMFrame0
  (:init
   (&key
    ((:id __id) 0)
    ((:system_time __system_time) 0)
    ((:pixel_count __pixel_count) 0)
    ((:pixels __pixels) ())
    )
   (send-super :init)
   (setq _id (round __id))
   (setq _system_time (round __system_time))
   (setq _pixel_count (round __pixel_count))
   (setq _pixels __pixels)
   self)
  (:id
   (&optional __id)
   (if __id (setq _id __id)) _id)
  (:system_time
   (&optional __system_time)
   (if __system_time (setq _system_time __system_time)) _system_time)
  (:pixel_count
   (&optional __pixel_count)
   (if __pixel_count (setq _pixel_count __pixel_count)) _pixel_count)
  (:pixels
   (&rest __pixels)
   (if (keywordp (car __pixels))
       (send* _pixels __pixels)
     (progn
       (if __pixels (setq _pixels (car __pixels)))
       _pixels)))
  (:serialization-length
   ()
   (+
    ;; uint8 _id
    1
    ;; uint32 _system_time
    4
    ;; uint8 _pixel_count
    1
    ;; nlink_parser/TofsenseMFrame0Pixel[] _pixels
    (apply #'+ (send-all _pixels :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _id
       (write-byte _id s)
     ;; uint32 _system_time
       (write-long _system_time s)
     ;; uint8 _pixel_count
       (write-byte _pixel_count s)
     ;; nlink_parser/TofsenseMFrame0Pixel[] _pixels
     (write-long (length _pixels) s)
     (dolist (elem _pixels)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _id
     (setq _id (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint32 _system_time
     (setq _system_time (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint8 _pixel_count
     (setq _pixel_count (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; nlink_parser/TofsenseMFrame0Pixel[] _pixels
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _pixels (let (r) (dotimes (i n) (push (instance nlink_parser::TofsenseMFrame0Pixel :init) r)) r))
     (dolist (elem- _pixels)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get nlink_parser::TofsenseMFrame0 :md5sum-) "8b49d724777a20439f181e513a9f9e83")
(setf (get nlink_parser::TofsenseMFrame0 :datatype-) "nlink_parser/TofsenseMFrame0")
(setf (get nlink_parser::TofsenseMFrame0 :definition-)
      "uint8 id
uint32 system_time
uint8 pixel_count
TofsenseMFrame0Pixel[] pixels

================================================================================
MSG: nlink_parser/TofsenseMFrame0Pixel
float32 dis
uint8 dis_status
uint16 signal_strength

")



(provide :nlink_parser/TofsenseMFrame0 "8b49d724777a20439f181e513a9f9e83")


