import cv2 as cv
print(cv.getBuildInformation())


# camera_backends = cv.videoio_registry.getCameraBackends()
# print(camera_backends)
# print([
#     cv.videoio_registry.getBackendName(apipref)
#     for apipref in camera_backends
# ])