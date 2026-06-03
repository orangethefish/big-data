variable "IMAGE_NAME" {
  default = "harmful-video-detection"
}

target "app" {
  context = "."
  dockerfile = "Dockerfile"
  tags = [
    "${IMAGE_NAME}:latest",
  ]
  platforms = [
    "linux/amd64",
    "linux/arm64",
  ]
}

group "default" {
  targets = ["app"]
}
