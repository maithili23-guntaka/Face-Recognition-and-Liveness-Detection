const video = document.getElementById('video');
const enrollButton = document.getElementById('enroll');
const recognizeButton = document.getElementById('recognize');
const exitButton = document.getElementById('exit');
const statusDiv = document.getElementById('status');

let labeledDescriptors = [];
let blinked = false;
let headMoved = false;
let livenessPassed = false;

const EAR_THRESHOLD = 0.25;
const HEAD_MOVE_THRESHOLD = 20;
let initialNoseX = null;

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('./models')
]).then(startVideo).then(() => {
  statusDiv.textContent = 'Models loaded. Ready to enroll or recognize.';
});

function startVideo() {
  return navigator.mediaDevices.getUserMedia({ video: {} })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error(err));
}

video.addEventListener('playing', () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detection = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

    if (detection) {
      const resized = faceapi.resizeResults(detection, displaySize);
      faceapi.draw.drawDetections(canvas, resized);
      faceapi.draw.drawFaceLandmarks(canvas, resized);

      const leftEye = resized.landmarks.getLeftEye();
      const rightEye = resized.landmarks.getRightEye();
      const ear = getEAR(leftEye, rightEye);

      if (ear < EAR_THRESHOLD && !blinked) {
        blinked = true;
        statusDiv.textContent = 'Blink detected.';
      }

      const nose = resized.landmarks.getNose();
      const noseX = nose[3].x;

      if (initialNoseX === null) {
        initialNoseX = noseX;
      } else if (Math.abs(noseX - initialNoseX) > HEAD_MOVE_THRESHOLD && !headMoved) {
        headMoved = true;
        statusDiv.textContent += ' Head movement detected.';
      }

      if (blinked && headMoved) {
        livenessPassed = true;
        statusDiv.textContent = '✅ Liveness confirmed (blink + head movement)';
      }
    }
  }, 300);
});

enrollButton.onclick = async () => {
  const descriptor = await getFaceDescriptor();
  if (descriptor) {
    const label = prompt('Enter your name:');
    labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, [descriptor]));
    statusDiv.textContent = `✅ Face enrolled for ${label}.`;
  } else {
    statusDiv.textContent = '❌ Face not detected.';
  }
};

recognizeButton.onclick = async () => {
  if (!livenessPassed) {
    statusDiv.textContent = '⚠️ Liveness check failed. Blink and move your head.';
    return;
  }
  const descriptor = await getFaceDescriptor();
  if (descriptor && labeledDescriptors.length > 0) {
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);
    const bestMatch = faceMatcher.findBestMatch(descriptor);
    statusDiv.textContent = `🔍 Recognized: ${bestMatch.toString()}`;
  } else {
    statusDiv.textContent = '❌ No enrolled faces or face not detected.';
  }
};

exitButton.onclick = () => {
  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  statusDiv.textContent = '🛑 Camera stopped.';
  blinked = false;
  headMoved = false;
  livenessPassed = false;
  initialNoseX = null;
};

function getEAR(eye1, eye2) {
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const leftEAR = (dist(eye1[1], eye1[5]) + dist(eye1[2], eye1[4])) / (2.0 * dist(eye1[0], eye1[3]));
  const rightEAR = (dist(eye2[1], eye2[5]) + dist(eye2[2], eye2[4])) / (2.0 * dist(eye2[0], eye2[3]));
  return (leftEAR + rightEAR) / 2.0;
}

async function getFaceDescriptor() {
  const detection = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks().withFaceDescriptor();
  return detection ? detection.descriptor : null;
}
