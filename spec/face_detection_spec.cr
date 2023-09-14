require "./spec_helper"

SPEC_FACE_IMAGE = Path.new "./bin/face_image.jpg"
SPEC_FACE_MODEL = Path.new "./bin/face_detection_back.tflite"

unless File.exists? SPEC_FACE_IMAGE
  puts "downloading image file for face spec..."
  HTTP::Client.get("https://aca.im/downloads/group-photo.jpeg") do |response|
    raise "could not download test image file" unless response.success?
    File.write(SPEC_FACE_IMAGE, response.body_io)
  end
end

unless File.exists? SPEC_FACE_MODEL
  puts "downloading tensorflow model for face spec..."
  # details: https://github.com/patlevin/face-detection-tflite
  # https://github.com/patlevin/face-detection-tflite/blob/main/fdlite/face_detection.py#L58
  HTTP::Client.get("https://raw.githubusercontent.com/patlevin/face-detection-tflite/main/fdlite/data/face_detection_back.tflite") do |response|
    raise "could not download tf model file" unless response.success?
    File.write(SPEC_FACE_MODEL, response.body_io)
  end
end

module TensorflowLite::Image
  describe FaceDetection do
    client = TensorflowLite::Client.new(SPEC_FACE_MODEL)
    detector = FaceDetection.new(client)

    # https://github.com/patlevin/face-detection-tflite/blob/main/fdlite/face_detection.py#L58
    detector.generate_anchors(strides: [16, 32, 32, 32])

    it "detects objects in an using fit image" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      puts "Anchor size: #{detector.anchors.size}"

      canvas = StumpyJPEG.read(SPEC_FACE_IMAGE.expand.to_s)
      scaled_canvas, detections = detector.run canvas
      puts detections.inspect

      detector.markup scaled_canvas, detections
      StumpyPNG.write(scaled_canvas, "./bin/face_fit_scaled_output.png")

      offsets = detector.detection_adjustments(canvas)
      detector.markup canvas, detections, *offsets
      StumpyPNG.write(canvas, "./bin/face_fit_original_output.png")
    end

    it "detects objects in an using cover image" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_FACE_IMAGE.expand.to_s)
      scaled_canvas, detections = detector.run canvas, scale_mode: :cover
      puts detections.inspect

      detector.markup scaled_canvas, detections
      StumpyPNG.write(scaled_canvas, "./bin/face_cover_scaled_output.png")

      offsets = detector.detection_adjustments(canvas, scale_mode: :cover)
      detector.markup canvas, detections, *offsets
      StumpyPNG.write(canvas, "./bin/face_cover_original_output.png")
    end
  end
end
