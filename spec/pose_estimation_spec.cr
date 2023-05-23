require "./spec_helper"

SPEC_POSE_IMAGE = Path.new "./bin/pose_image.jpg"
SPEC_POSE_MODEL = Path.new "./bin/pose_movenet.tflite"

unless File.exists? SPEC_POSE_IMAGE
  puts "downloading image file for classification spec..."
  HTTP::Client.get("https://aca.im/downloads/pose.jpg") do |response|
    raise "could not download test image file" unless response.success?
    File.write(SPEC_POSE_IMAGE, response.body_io)
  end
end

unless File.exists? SPEC_POSE_MODEL
  puts "downloading tensorflow model for classification spec..."
  # details: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4
  HTTP::Client.get("https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/int8/4.tflite") do |response|
    raise "could not download tf model file" unless response.success?
    File.write(SPEC_POSE_MODEL, response.body_io)
  end
end

module TensorflowLite::Image
  describe PoseEstimation do
    client = TensorflowLite::Client.new(SPEC_POSE_MODEL)
    pose = PoseEstimation.new(client)

    it "detects poses in an image" do
      puts client.interpreter.inspect
      puts "input resolution: #{pose.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_POSE_IMAGE.expand.to_s)
      scaled_canvas, detections = pose.run canvas, scale_mode: :cover
      detections.size.should eq 17

      pose.markup scaled_canvas, detections
      StumpyPNG.write(scaled_canvas, "./bin/poses_cover_scaled_output.png")

      offsets = pose.detection_adjustments(canvas, scale_mode: :cover)
      pose.markup canvas, detections, *offsets
      StumpyPNG.write(canvas, "./bin/poses_cover_original_output.png")
    end
  end
end
