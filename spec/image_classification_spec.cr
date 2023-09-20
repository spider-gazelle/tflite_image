require "./spec_helper"

SPEC_CLASSIFY_IMAGE = Path.new "./bin/class_image.jpg"
SPEC_CLASSIFY_MODEL = Path.new "./bin/class_mobilenet_v2.tflite"

unless File.exists? SPEC_CLASSIFY_IMAGE
  puts "downloading image file for classification spec..."
  HTTP::Client.get("https://aca.im/downloads/apple.jpg") do |response|
    raise "could not download test image file" unless response.success?
    File.write(SPEC_CLASSIFY_IMAGE, response.body_io)
  end
end

unless File.exists? SPEC_CLASSIFY_MODEL
  puts "downloading tensorflow model for classification spec..."
  # details: https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224_quantized/1/metadata/1
  HTTP::Client.get("https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/mobilenet_v2_1.0_224_quantized/1/metadata/1.tflite") do |response|
    raise "could not download tf model file" unless response.success?
    File.write(SPEC_CLASSIFY_MODEL, response.body_io)
  end
end

describe TensorflowLite::Image::Classification do
  client = TensorflowLite::Client.new(SPEC_CLASSIFY_MODEL)
  classifier = TensorflowLite::Image::Classification.new(client)

  it "classifies an image" do
    puts client.interpreter.inspect
    puts "input resolution: #{classifier.resolution.join("x")}px"

    canvas = StumpyJPEG.read(SPEC_CLASSIFY_IMAGE.expand.to_s)
    scaled_canvas, detections = classifier.run canvas
    puts detections.inspect

    StumpyPNG.write(scaled_canvas, "./bin/classification_output.png")

    detections[0].label.should eq "pomegranate"
    detections[1].label.should eq "Granny Smith"
  end
end
