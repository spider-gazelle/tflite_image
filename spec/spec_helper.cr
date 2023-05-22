require "http"
require "spec"
require "stumpy_png"
require "stumpy_jpeg"
require "../src/tflite_image"

Spec.before_suite do
  Log.setup(:trace)
end

Dir.mkdir_p "./bin/"
