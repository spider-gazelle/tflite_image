require "benchmark"

require "http"
require "spec"
require "stumpy_png"
require "stumpy_jpeg"
require "./src/tflite_image"
require "tensorflow_lite/edge_tpu"

# is there an Edge TPU available?
delegate = if TensorflowLite::EdgeTPU.devices.size > 0
                 edge_tpu = TensorflowLite::EdgeTPU.devices[0]
                 Log.info { "EdgeTPU Found! #{edge_tpu.type}: #{edge_tpu.path}" }
                 edge_tpu.to_delegate
               end

# init the tensorflow lite library for a TPU
client = TensorflowLite::Client.new(
  model: URI.parse("https://raw.githubusercontent.com/google-coral/test_data/master/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"),
  labels: URI.parse("https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt"),
  delegate: delegate
)

# prepare for object detection
detector = TensorflowLite::Image::ObjectDetection.new(client)
desired_width, desired_height = detector.resolution

# scale an image to the input size
canvas = StumpyJPEG.read(Path.new("./bin/detect_image.jpg").expand.to_s)
scaled = StumpyResize.scale_to_cover(canvas, desired_width, desired_height)

# warm up the model
scaled_canvas, detections = detector.run scaled
puts detections.inspect

# init the tensorflow lite library for a CPU model
cpu_client = TensorflowLite::Client.new(
  model: URI.parse("https://raw.githubusercontent.com/google-coral/test_data/master/ssdlite_mobiledet_coco_qat_postprocess.tflite"),
  labels: URI.parse("https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt")
)

# prepare for object detection
cpu_detector = TensorflowLite::Image::ObjectDetection.new(cpu_client)

Benchmark.ips do |x|
  x.report("TPU detections per second: ") { detector.run scaled }
  x.report("CPU detections per second: ") { cpu_detector.run scaled }
  x.report("NN resizing a second") { StumpyResize.scale_to_cover(canvas, desired_width, desired_height, :nearest_neighbor) }
  x.report("Bilinear resizing a second") { StumpyResize.scale_to_cover(canvas, desired_width, desired_height, :bilinear) }
end
