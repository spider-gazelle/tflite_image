require "./spec_helper"

SPEC_DETECT_IMAGE = Path.new "./bin/detect_image.jpg"
SPEC_DETECT_MODEL = Path.new "./bin/detect_efficientdet.tflite"

unless File.exists? SPEC_DETECT_IMAGE
  puts "downloading image file for detection spec..."
  HTTP::Client.get("https://aca.im/downloads/kitchen.jpg") do |response|
    raise "could not download test image file" unless response.success?
    File.write(SPEC_DETECT_IMAGE, response.body_io)
  end
end

unless File.exists? SPEC_DETECT_MODEL
  puts "downloading tensorflow model for detection spec..."
  # details: https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1
  HTTP::Client.get("https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1.tflite") do |response|
    raise "could not download tf model file" unless response.success?
    File.write(SPEC_DETECT_MODEL, response.body_io)
  end
end

module TensorflowLite::Image
  describe ObjectDetection do
    client = TensorflowLite::Client.new(SPEC_DETECT_MODEL)
    detector = ObjectDetection.new(client)

    it "detects objects in an using fit image" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_DETECT_IMAGE.expand.to_s)
      scaled_canvas, detections = detector.run canvas
      puts detections.inspect

      Image.markup scaled_canvas, detections
      StumpyPNG.write(scaled_canvas, "./bin/detection_fit_scaled_output.png")

      offsets = detector.detection_adjustments(canvas)
      Image.adjust detections, canvas, *offsets
      Image.markup canvas, detections
      StumpyPNG.write(canvas, "./bin/detection_fit_original_output.png")

      detections[0].label.should eq "dining table"
    end

    it "detects objects in an using cover image" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_DETECT_IMAGE.expand.to_s)
      scaled_canvas, detections = detector.run canvas, scale_mode: :cover
      puts detections.inspect

      Image.markup scaled_canvas, detections
      StumpyPNG.write(scaled_canvas, "./bin/detection_cover_scaled_output.png")

      offsets = detector.detection_adjustments(canvas, scale_mode: :cover)
      Image.adjust detections, canvas, *offsets
      Image.markup canvas, detections
      StumpyPNG.write(canvas, "./bin/detection_cover_original_output.png")

      detections[0].label.should eq "potted plant"
    end

    it "can adjust bounding boxes to fit a particular aspect ratio" do
      # matching aspect ratio
      detection = ObjectDetection::Output.new(top: 0.0, left: 0.0, bottom: 0.03, right: 0.03, score: 0.0, index: 1)
      position = detection.adjust_bounding_box(4 / 4, 100, 100)
      position[:top].should eq 0
      position[:left].should eq 0
      position[:bottom].should eq 3
      position[:right].should eq 3

      # aspect ratio less than target, against edge
      detection = ObjectDetection::Output.new(top: 0.0, left: 0.0, bottom: 0.06, right: 0.06, score: 0.0, index: 1)
      position = detection.adjust_bounding_box(4 / 3, 100, 100)
      position[:top].should eq 0
      position[:left].should eq 0
      position[:bottom].should eq 6
      position[:right].should eq 8

      # aspect ratio less than target, centered
      detection = ObjectDetection::Output.new(top: 0.02, left: 0.02, bottom: 0.08, right: 0.08, score: 0.0, index: 1)
      position = detection.adjust_bounding_box(4 / 3, 100, 100)
      position[:top].should eq 2
      position[:left].should eq 1
      position[:bottom].should eq 8
      position[:right].should eq 9

      # aspect ratio less than target, far edge
      detection = ObjectDetection::Output.new(top: 0.94, left: 0.94, bottom: 1.0, right: 1.0, score: 0.0, index: 1)
      position = detection.adjust_bounding_box(4 / 3, 100, 100)
      position[:top].should eq 94
      position[:left].should eq 92
      position[:bottom].should eq 100
      position[:right].should eq 100

      # aspect ratio cannot be matched
      detection = ObjectDetection::Output.new(top: 1.0, left: 1.0, bottom: 1.0, right: 1.0, score: 0.0, index: 1)
      position = detection.adjust_bounding_box(4 / 3, 100, 100)
      position[:top].should eq 100
      position[:left].should eq 100
      position[:bottom].should eq 100
      position[:right].should eq 100

      # aspect ratio greater than target, against edge
      detection = ObjectDetection::Output.new(top: 0.0, left: 0.0, bottom: 0.06, right: 0.06, score: 0.0, index: 1)
      position = detection.adjust_bounding_box(3 / 4, 100, 100)
      position[:top].should eq 0
      position[:left].should eq 0
      position[:bottom].should eq 8
      position[:right].should eq 6

      # aspect ratio greater than target, centered
      detection = ObjectDetection::Output.new(top: 0.02, left: 0.02, bottom: 0.08, right: 0.08, score: 0.0, index: 1)
      position = detection.adjust_bounding_box(3 / 4, 100, 100)
      position[:top].should eq 1
      position[:left].should eq 2
      position[:bottom].should eq 9
      position[:right].should eq 8

      # aspect ratio greater than target, far edge
      detection = ObjectDetection::Output.new(top: 0.94, left: 0.94, bottom: 1.0, right: 1.0, score: 0.0, index: 1)
      position = detection.adjust_bounding_box(3 / 4, 100, 100)
      position[:top].should eq 92
      position[:left].should eq 94
      position[:bottom].should eq 100
      position[:right].should eq 100
    end
  end
end
