async function initializeTensorFlow() {
  await tf.ready();
}

async function loadModelAndDetect() {
  const detectorConfig = {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
  };

  const detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    detectorConfig
  );

  const videoElement = document.getElementById("webcam-video");
  const canvasElement = document.getElementById("pose-canvas");
  const pushupStageElement = document.getElementById("pushup-stage");
  const situpStageElement = document.getElementById("situp-stage");
  const pushupCountElement = document.getElementById("pushup-count");
  const situpCountElement = document.getElementById("situp-count");

  const detectPoses = async () => {
    const poses = await detector.estimatePoses(videoElement);

    // Check if any poses are detected
    if (poses.length > 0) {
      const pose = poses[0];
      const inputList = [];
      pose.keypoints.forEach((keypoint) => {
        inputList.push(keypoint.x);
        inputList.push(keypoint.y);
      });

      const PUSHUP_MODEL_URL = "/pushup/model.json";
      const SITUP_MODEL_URL = "/situp/model.json";

      const pushup_model = await tf.loadLayersModel(PUSHUP_MODEL_URL);
      const situp_model = await tf.loadLayersModel(SITUP_MODEL_URL);
      const inputTensor = tf.tensor2d(inputList, [1, inputList.length]);

      if (pose.score > 0.5) {
        const pushup_prediction = pushup_model
          .predict(inputTensor)
          .dataSync()[0];
        const situp_prediction = situp_model.predict(inputTensor).dataSync()[0];
        console.log("Pushup Prediction:", pushup_prediction);
        console.log("Situp Prediction:", situp_prediction);

        if (pushup_prediction < 0.1) {
          pushupStageElement.textContent = "down";
        }
        if (
          pushup_prediction > 0.9 &&
          pushupStageElement.textContent === "down"
        ) {
          const prevCount = parseInt(pushupCountElement.textContent);
          pushupCountElement.textContent = prevCount + 1;
          pushupStageElement.textContent = "up";
        }

        if (
          situp_prediction > 0.9 &&
          situpStageElement.textContent === "down"
        ) {
          const prevCount = parseInt(situpCountElement.textContent);
          situpCountElement.textContent = prevCount + 1;
          situpStageElement.textContent = "up";
        }
        if (situp_prediction < 0.1 && situpStageElement.textContent === "up") {
          situpStageElement.textContent = "down";
        }
      }

      // Don't forget to dispose the inputTensor to free up memory
      inputTensor.dispose();

      // Draw poses on canvas (example)
      const ctx = canvasElement.getContext("2d");
      ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      poses.forEach(({ keypoints }) => {
        keypoints.forEach(({ x, y }) => {
          // Adjust x-coordinate to center keypoints
          const adjustedX = (x + 480) / 2;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "red";
          ctx.fill();
        });
      });
    }

    // RequestAnimationFrame for the next frame
    requestAnimationFrame(detectPoses);
  };

  // Start detecting poses
  detectPoses();
}

function handleVideo(stream) {
  const videoElement = document.getElementById("webcam-video");

  // Set the video stream as the source for the video element
  videoElement.srcObject = stream;
}

function videoError(error) {
  console.error("Error accessing webcam:", error);
}

document.addEventListener("DOMContentLoaded", async () => {
  await initializeTensorFlow(); // Initialize TensorFlow

  const constraints = { video: true };

  try {
    // Request access to the webcam stream
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    handleVideo(stream);
  } catch (error) {
    console.error("Error accessing webcam:", error);
  }

  const videoElement = document.getElementById("webcam-video");
  const canvasElement = document.getElementById("pose-canvas");

  videoElement.onloadedmetadata = () => {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  };

  loadModelAndDetect();
});
