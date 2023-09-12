require "./spec_helper"
require "./gender_estimation_spec"

SPEC_AGE_MODEL       = Path.new "./bin/age_detection.tflite"
SPEC_AGE_RANGE_MODEL = Path.new "./bin/age_range.tflite"
SPEC_AGEM_IMAGE      = Path.new "./bin/old_man.jpg"
SPEC_AGEW_IMAGE      = Path.new "./bin/old_woman.jpg"

unless File.exists? SPEC_AGEM_IMAGE
  puts "downloading image file for gender spec..."
  HTTP::Client.get("https://os.place.tech/neural_nets/age/old_man.jpg") do |response|
    raise "could not download test image file" unless response.success?
    File.write(SPEC_AGEM_IMAGE, response.body_io)
  end
end

unless File.exists? SPEC_AGEW_IMAGE
  puts "downloading image file for gender spec..."
  HTTP::Client.get("https://os.place.tech/neural_nets/age/old_woman.jpg") do |response|
    raise "could not download test image file" unless response.success?
    File.write(SPEC_AGEW_IMAGE, response.body_io)
  end
end

unless File.exists? SPEC_AGE_MODEL
  puts "downloading tensorflow model for gender spec..."
  # details: https://github.com/shubham0204/Age-Gender_Estimation_TF-Android
  HTTP::Client.get("https://os.place.tech/neural_nets/age/model_age_nonq.tflite") do |response|
    raise "could not download tf model file" unless response.success?
    File.write(SPEC_AGE_MODEL, response.body_io)
  end
end

unless File.exists? SPEC_AGE_RANGE_MODEL
  puts "downloading tensorflow model for gender spec..."
  # details: https://github.com/radualexandrub/Age-Gender-Classification-on-RaspberryPi4-with-TFLite-PyQt5
  HTTP::Client.get("https://os.place.tech/neural_nets/age/age_range.tflite") do |response|
    raise "could not download tf model file" unless response.success?
    File.write(SPEC_AGE_RANGE_MODEL, response.body_io)
  end
end

module TensorflowLite::Image
  describe AgeEstimationExact do
    client = TensorflowLite::Client.new(SPEC_AGE_MODEL)
    detector = AgeEstimationExact.new(client)

    it "detects young male age" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_GENDER_MALE_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      puts detection.inspect
    end

    it "detects young female age" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_GENDER_FEMALE_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      puts detection.inspect
    end

    it "detects old male age" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_AGEM_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      puts detection.inspect
    end

    it "detects old female age" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_AGEW_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      puts detection.inspect
    end
  end

  describe AgeEstimationRange do
    client = TensorflowLite::Client.new(SPEC_AGE_RANGE_MODEL)
    detector = AgeEstimationRange.new(client)

    it "detects young male age" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_GENDER_MALE_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      detection.age_estimate
      puts detection.inspect
      detection.age_estimate.includes?(25).should be_true
    end

    it "detects young female age" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_GENDER_FEMALE_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      detection.age_estimate
      puts detection.inspect
      detection.age_estimate.includes?(20).should be_true
    end

    it "detects old male age" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_AGEM_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      detection.age_estimate
      puts detection.inspect
      detection.age_estimate.includes?(60).should be_true
    end

    it "detects old female age" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_AGEW_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      detection.age_estimate
      puts detection.inspect
      detection.age_estimate.includes?(65).should be_true
    end
  end
end
