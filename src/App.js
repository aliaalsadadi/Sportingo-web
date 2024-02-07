import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
async function initializeTensorFlow() {
  await tf.ready();
}

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [canvasDimensions, setCanvasDimensions] = useState({
    width: 640,
    height: 480,
  });
  const [pushupStage, setPushupStage] = useState("down");
  const [situpStage, setSitupStage] = useState("down");
  const [prevPushupStage, setPrevPushupStage] = useState("down");
  const [pushupCount, setPushupCount] = useState(0);
  const [situpCount, setSitupCount] = useState(0);

  useEffect(() => {
    initializeTensorFlow(); // Initialize TensorFlow

    const detectorConfig = {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
    };

    async function loadModelAndDetect() {
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        detectorConfig
      );

      const detectPoses = async () => {
        if (
          typeof webcamRef.current !== "undefined" &&
          webcamRef.current !== null &&
          webcamRef.current.video.readyState === 4
        ) {
          const video = webcamRef.current.video;

          const poses = await detector.estimatePoses(video);

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
              const situp_prediction = situp_model
                .predict(inputTensor)
                .dataSync()[0];
              console.log("Pushup Prediction:", pushup_prediction);
              console.log("Situp Prediction:", situp_prediction);

              if (pushup_prediction < 0.1) {
                setPushupStage("down");
              }
              if (pushup_prediction > 0.9 && pushupStage === "down") {
                setPushupCount((prevCount) => prevCount + 1);
                setPushupStage("up");
              }

              // Update the previous pushup stage
              setPrevPushupStage(pushupStage);

              if (situp_prediction > 0.9 && situpStage === "down") {
                setSitupStage("up");
                setSitupCount((prevCount) => prevCount + 1);
              }
              if (situp_prediction < 0.1 && situpStage === "up") {
                setSitupStage("down");
              }
            }

            // Don't forget to dispose the inputTensor to free up memory
            inputTensor.dispose();

            // Draw poses on canvas (example)
            // Draw poses on canvas
            const ctx = canvasRef.current.getContext("2d");
            ctx.clearRect(
              0,
              0,
              canvasDimensions.width,
              canvasDimensions.height
            );
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
        }

        // Request the next frame
        requestAnimationFrame(detectPoses);
      };

      // Start detecting poses
      detectPoses();
    }

    loadModelAndDetect();
  }, []);

  const handleVideoLoaded = () => {
    const videoElement = webcamRef.current.video;
    videoElement.onloadedmetadata = () => {
      setCanvasDimensions({
        width: videoElement.videoWidth,
        height: videoElement.videoHeight,
      });
    };
  };

  return (
    <div className="App">
      <header className="App-header">
        <div style={{ position: "relative" }}>
          <Webcam
            style={{ display: "block", margin: "auto" }}
            ref={webcamRef}
            onUserMedia={handleVideoLoaded}
          />
          <canvas
            ref={canvasRef}
            width={canvasDimensions.width}
            height={canvasDimensions.height}
            style={{ position: "absolute", top: 0, left: 0 }}
          ></canvas>
          <div>
            <p>Pushup Stage: {pushupStage}</p>
            <p>Situp Stage: {situpStage}</p>
            <p>Pushup Count: {pushupCount}</p>
            <p>Situp Count: {situpCount}</p>
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
