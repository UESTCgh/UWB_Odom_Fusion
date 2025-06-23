;; Auto-generated. Do not edit!


(when (boundp 'nlink_parser::IotFrame0Node)
  (if (not (find-package "NLINK_PARSER"))
    (make-package "NLINK_PARSER"))
  (shadow 'IotFrame0Node (find-package "NLINK_PARSER")))
(unless (find-package "NLINK_PARSER::IOTFRAME0NODE")
  (make-package "NLINK_PARSER::IOTFRAME0NODE"))

(in-package "ROS")
;;//! \htmlinclude IotFrame0Node.msg.html


(defclass nlink_parser::IotFrame0Node
  :super ros::object
  :slots (_uid _dis _aoa_angle_horizontal _aoa_angle_vertical _fp_rssi _rx_rssi _user_data ))

(defmethod nlink_parser::IotFrame0Node
  (:init
   (&key
    ((:uid __uid) 0)
    ((:dis __dis) 0.0)
    ((:aoa_angle_horizontal __aoa_angle_horizontal) 0.0)
    ((:aoa_angle_vertical __aoa_angle_vertical) 0.0)
    ((:fp_rssi __fp_rssi) 0.0)
    ((:rx_rssi __rx_rssi) 0.0)
    ((:user_data __user_data) "")
    )
   (send-super :init)
   (setq _uid (round __uid))
   (setq _dis (float __dis))
   (setq _aoa_angle_horizontal (float __aoa_angle_horizontal))
   (setq _aoa_angle_vertical (float __aoa_angle_vertical))
   (setq _fp_rssi (float __fp_rssi))
   (setq _rx_rssi (float __rx_rssi))
   (setq _user_data (string __user_data))
   self)
  (:uid
   (&optional __uid)
   (if __uid (setq _uid __uid)) _uid)
  (:dis
   (&optional __dis)
   (if __dis (setq _dis __dis)) _dis)
  (:aoa_angle_horizontal
   (&optional __aoa_angle_horizontal)
   (if __aoa_angle_horizontal (setq _aoa_angle_horizontal __aoa_angle_horizontal)) _aoa_angle_horizontal)
  (:aoa_angle_vertical
   (&optional __aoa_angle_vertical)
   (if __aoa_angle_vertical (setq _aoa_angle_vertical __aoa_angle_vertical)) _aoa_angle_vertical)
  (:fp_rssi
   (&optional __fp_rssi)
   (if __fp_rssi (setq _fp_rssi __fp_rssi)) _fp_rssi)
  (:rx_rssi
   (&optional __rx_rssi)
   (if __rx_rssi (setq _rx_rssi __rx_rssi)) _rx_rssi)
  (:user_data
   (&optional __user_data)
   (if __user_data (setq _user_data __user_data)) _user_data)
  (:serialization-length
   ()
   (+
    ;; uint32 _uid
    4
    ;; float32 _dis
    4
    ;; float32 _aoa_angle_horizontal
    4
    ;; float32 _aoa_angle_vertical
    4
    ;; float32 _fp_rssi
    4
    ;; float32 _rx_rssi
    4
    ;; string _user_data
    4 (length _user_data)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint32 _uid
       (write-long _uid s)
     ;; float32 _dis
       (sys::poke _dis (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _aoa_angle_horizontal
       (sys::poke _aoa_angle_horizontal (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _aoa_angle_vertical
       (sys::poke _aoa_angle_vertical (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _fp_rssi
       (sys::poke _fp_rssi (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rx_rssi
       (sys::poke _rx_rssi (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; string _user_data
       (write-long (length _user_data) s) (princ _user_data s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint32 _uid
     (setq _uid (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32 _dis
     (setq _dis (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _aoa_angle_horizontal
     (setq _aoa_angle_horizontal (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _aoa_angle_vertical
     (setq _aoa_angle_vertical (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _fp_rssi
     (setq _fp_rssi (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rx_rssi
     (setq _rx_rssi (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; string _user_data
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _user_data (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get nlink_parser::IotFrame0Node :md5sum-) "df423b5fa843bedeae14cfe493476452")
(setf (get nlink_parser::IotFrame0Node :datatype-) "nlink_parser/IotFrame0Node")
(setf (get nlink_parser::IotFrame0Node :definition-)
      "uint32 uid
float32 dis
float32 aoa_angle_horizontal
float32 aoa_angle_vertical
float32 fp_rssi
float32 rx_rssi
string user_data

")



(provide :nlink_parser/IotFrame0Node "df423b5fa843bedeae14cfe493476452")


