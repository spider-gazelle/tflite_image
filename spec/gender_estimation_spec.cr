require "./spec_helper"

SPEC_GENDER_FEMALE_IMAGE = Path.new "./bin/gender_female.jpg"
SPEC_GENDER_MALE_IMAGE   = Path.new "./bin/gender_male.jpg"
SPEC_GENDER_MODEL        = Path.new "./bin/gender_detection.tflite"

unless File.exists? SPEC_GENDER_FEMALE_IMAGE
  puts "downloading image file for gender spec..."
  HTTP::Client.get("https://os.place.tech/neural_nets/gender/face3.jpg") do |response|
    raise "could not download test image file" unless response.success?
    File.write(SPEC_GENDER_FEMALE_IMAGE, response.body_io)
  end
end

unless File.exists? SPEC_GENDER_MALE_IMAGE
  puts "downloading image file for gender spec..."
  HTTP::Client.get("https://os.place.tech/neural_nets/gender/face.jpeg") do |response|
    raise "could not download test image file" unless response.success?
    File.write(SPEC_GENDER_MALE_IMAGE, response.body_io)
  end
end

unless File.exists? SPEC_GENDER_MODEL
  puts "downloading tensorflow model for gender spec..."
  # details: https://github.com/shubham0204/Age-Gender_Estimation_TF-Android
  HTTP::Client.get("https://os.place.tech/neural_nets/gender/model_lite_gender_q.tflite") do |response|
    raise "could not download tf model file" unless response.success?
    File.write(SPEC_GENDER_MODEL, response.body_io)
  end
end

module TensorflowLite::Image
  describe GenderEstimation do
    client = TensorflowLite::Client.new(SPEC_GENDER_MODEL)
    detector = GenderEstimation.new(client)

    it "detects the male gender" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_GENDER_MALE_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      detection.gender
      puts detection.inspect

      detection.gender.male?.should eq true
    end

    it "detects the female gender" do
      puts client.interpreter.inspect
      puts "input resolution: #{detector.resolution.join("x")}px"

      canvas = StumpyJPEG.read(SPEC_GENDER_FEMALE_IMAGE.expand.to_s)
      _, detections = detector.run canvas

      # set the gender variable
      detection = detections[0]
      detection.gender
      puts detection.inspect

      detection.gender.female?.should eq true
    end
  end
end
