import * as THREE from "three";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

/* =========================
   Utils / Math
========================= */
function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
function dist3(a,b){
  const dx=a.x-b.x, dy=a.y-b.y, dz=(a.z||0)-(b.z||0);
  return Math.sqrt(dx*dx+dy*dy+dz*dz);
}

/* =========================
   EMA (сглаживание)
========================= */
class EMA {
  constructor(alpha=0.25, size=5){
    this.alpha=alpha;
    this.value=new Array(size).fill(0);
    this.init=false;
  }
  update(cur01){
    if(!this.init){
      this.value = cur01.map(x=>+x);
      this.init=true;
      return this.value.slice();
    }
    for(let i=0;i<cur01.length;i++){
      this.value[i] = this.alpha*cur01[i] + (1-this.alpha)*this.value[i];
    }
    return this.value.slice();
  }
}

class EMA2 {
  constructor(alpha=0.35){
    this.alpha=alpha;
    this.x=0; this.y=0; this.init=false;
  }
  update(dx,dy){
    if(!this.init){ this.x=dx; this.y=dy; this.init=true; }
    else{
      this.x = this.alpha*dx + (1-this.alpha)*this.x;
      this.y = this.alpha*dy + (1-this.alpha)*this.y;
    }
    return {dx:this.x, dy:this.y};
  }
}

/* =========================
   Gesture FSM (подтверждение/снятие)
========================= */
class GestureFSM {
  constructor(confirmFrames=5, releaseFrames=3){
    this.confirmFrames=confirmFrames;
    this.releaseFrames=releaseFrames;
    this.current=null;
    this.confirm=0;
    this.release=0;
    this.active=false;
  }
  update(nameOrNull){
    // есть жест
    if(nameOrNull){
      this.release=0;
      if(nameOrNull !== this.current){
        this.current = nameOrNull;
        this.confirm = 1;
        this.active = false;
        return null;
      }
      if(!this.active){
        this.confirm++;
        if(this.confirm >= this.confirmFrames){
          this.active=true;
          return {type:"START", name:this.current};
        }
      }
      return null;
    }
    // нет жеста
    this.confirm=0;
    if(!this.current) return null;
    this.release++;
    if(this.active && this.release >= this.releaseFrames){
      const ended=this.current;
      this.current=null; this.active=false; this.release=0;
      return {type:"END", name:ended};
    }
    if(!this.active && this.release >= this.releaseFrames){
      this.current=null; this.release=0;
    }
    return null;
  }
}

/* =========================
   HandState (как в Python)
   landmarks: array of 21 points with {x,y,z}
========================= */
class HandState{
  constructor(landmarks, handedness){
    this.landmarks = landmarks;
    this.handedness = handedness;
    this.thumbTip = landmarks[4];
    this.indexTip = landmarks[8];
    this.palmCenter = this.computePalmCenter();
    this.pinchDistance = this.dist3(this.thumbTip, this.indexTip);
  }

  dist3(a,b){
    const dx=a.x-b.x, dy=a.y-b.y, dz=(a.z-b.z);
    return Math.sqrt(dx*dx+dy*dy+dz*dz);
  }

  computePalmCenter(){
    const ids=[0,5,9,13,17];
    let x=0,y=0,z=0;
    for(const i of ids){ x+=this.landmarks[i].x; y+=this.landmarks[i].y; z+=this.landmarks[i].z; }
    return {x:x/ids.length, y:y/ids.length, z:z/ids.length};
  }
}


/* =========================
   FingerAnalyzer (finger states + EMA)
   Здесь сделаем упрощённую модель:
   - 4 пальца: tip.y < pip.y  (в поднятом состоянии)
   - большой: по x с учётом handedness
========================= */
class FingerAnalyzer {
  constructor(){
    this.ema = new EMA(0.25, 5);
  }
  rawStates(landmarks, handedness){
    // индексы как в классике:
    const tips=[4,8,12,16,20];
    const pips=[3,6,10,14,18];

    const s=[0,0,0,0,0];

    const tip=landmarks[tips[0]];
    const ip =landmarks[pips[0]];
    if(handedness==="Right") s[0] = tip.x > ip.x ? 1:0;
    else if(handedness==="Left") s[0] = tip.x < ip.x ? 1:0;
    else s[0] = tip.x > ip.x ? 1:0;

    for(let i=1;i<5;i++){
      const t=landmarks[tips[i]];
      const p=landmarks[pips[i]];
      s[i] = (t.y < p.y) ? 1:0; // y вниз растёт
    }
    return s;
  }
  update(landmarks, handedness){
    const raw = this.rawStates(landmarks, handedness);
    const sm = this.ema.update(raw);
    const stable = sm.map(v => v>0.6 ? 1:0);
    return stable;
  }
}

/* =========================
   GestureDetector (форма руки)
========================= */
class GestureDetector {
  detect(stableStates){
    const [t,i,m,r,p]=stableStates;
    if(t===0 && i===0 && m===0 && r===0 && p===0) return "FIST";
    if(t===1 && i===1 && m===1 && r===1 && p===1) return "PALM";
    if(i===1 && m===1 && r===0 && p===0) return "V_SIGN";
    return null;
  }
}

/* =========================
   InteractionEngine
   - PINCH START/END (hysteresis)
   - DRAG while pinch: dx/dy по palmCenter
   - FSM for gesture forms when NOT pinch
========================= */
class InteractionEngine {
  constructor(){
    this.fsms = new Map();        // handId -> GestureFSM
    this.pinch = new Map();       // handId -> bool
    this.prevPalm = new Map();    // handId -> {x,y}
    this.dragEma = new Map();     // handId -> EMA2

    this.PINCH_ENTER = 0.035;
    this.PINCH_EXIT  = 0.045;
    this.DEADZONE = 0.0015;
  }
  ensure(handId){
    if(!this.fsms.has(handId)) this.fsms.set(handId, new GestureFSM(5,3));
    if(!this.pinch.has(handId)) this.pinch.set(handId,false);
    if(!this.dragEma.has(handId)) this.dragEma.set(handId, new EMA2(0.35));
  }
  isPinch(handId){ return !!this.pinch.get(handId); }

  update(handId, state, gestureName){
    this.ensure(handId);
    const events=[];
    const pinchActive = this.pinch.get(handId);

    // PINCH hysteresis
    if(!pinchActive && state.pinchDistance < this.PINCH_ENTER){
      this.pinch.set(handId,true);
      this.prevPalm.set(handId, {x:state.palmCenter.x, y:state.palmCenter.y});
      events.push({type:"START", name:"PINCH"});
    } else if(pinchActive && state.pinchDistance > this.PINCH_EXIT){
      this.pinch.set(handId,false);
      this.prevPalm.delete(handId);
      events.push({type:"END", name:"PINCH"});
    }

    // DRAG while pinch
    if(this.pinch.get(handId)){
      const prev = this.prevPalm.get(handId);
      const cur = {x:state.palmCenter.x, y:state.palmCenter.y};
      if(prev){
        const dx = cur.x - prev.x;
        const dy = cur.y - prev.y;
        const {dx:sdx, dy:sdy} = this.dragEma.get(handId).update(dx,dy);
        if(Math.abs(sdx) > this.DEADZONE || Math.abs(sdy) > this.DEADZONE){
          events.push({type:"DRAG", dx:sdx, dy:sdy});
        }
      }
      this.prevPalm.set(handId, cur);
      return events; // во время pinch не шлём жесты формы
    }

    // gesture form events (FSM)
    const fsm = this.fsms.get(handId);
    const ev = fsm.update(gestureName);
    if(ev) events.push({type:ev.type, name:ev.name});
    return events;
  }
}

/* =========================
   AR Scene (Three.js)
========================= */
class ARScene {
  constructor(canvas){
    this.canvas = canvas;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 100);
    this.camera.position.set(0, 0, 2);

    this.renderer = new THREE.WebGLRenderer({canvas, alpha:true, antialias:true});
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));

    const light = new THREE.DirectionalLight(0xffffff, 1.0);
    light.position.set(1, 1, 2);
    this.scene.add(light);

    const geo = new THREE.BoxGeometry(0.15,0.15,0.15);

    // материалы (норм / hover / grabbed)
    this.matNormal = new THREE.MeshStandardMaterial({color: 0x00ff88});
    this.matHover  = new THREE.MeshStandardMaterial({color: 0xffee00});
    this.matGrab   = new THREE.MeshStandardMaterial({color: 0xff00aa});

// ================= OBJECT SYSTEM =================
this.objects = [];
this.selectedObject = null;
this.selectionMode = false;
this.nextId = 1;

// материалы
this.matSelectable = new THREE.MeshStandardMaterial({color: 0x3399ff});
this.matSelected   = new THREE.MeshStandardMaterial({color: 0xaa00ff});


    // --- grab/hover ---
    this.isGrabbed = false;
    // ================= PHYSICS =================
this.velocity = new THREE.Vector3();
this.damping = 0.94;
this.isInertia = false;
this.lastFrameTime = performance.now();

    this.grabOffset = new THREE.Vector3();
    this.hovered = false;

    // Радиусы в "метрах" сцены (тюнится)
    this.hoverRadius = 0.28;
    this.grabRadius  = 0.22;

    // --- pinch zoom (Z) ---
    this.basePinchDist = null; // значение pinchDistance в момент захвата
    this.baseZ = 0;            // z куба в момент захвата
    this.zoomGain = 2.2;       // чувствительность zoom по Z (тюнится)
    this.zMin = -1.0;          // ограничения по Z
    this.zMax =  0.8;

    // ray helpers
    this.raycaster = new THREE.Raycaster();
    this.plane = new THREE.Plane(new THREE.Vector3(0,0,1), 0);

    window.addEventListener("resize", () => {
      this.camera.aspect = window.innerWidth/window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
    // ================= TWO HAND =================
this.isDual = false;
this.baseDistance = 0;
this.baseScale = 1;
this.baseAngle = 0;
this.baseRotation = 0;

  }
spawnObject(worldPos){

  const geo = new THREE.BoxGeometry(0.15,0.15,0.15);
  const mesh = new THREE.Mesh(geo, this.matNormal);

  mesh.position.copy(worldPos);

  const obj = {
    id: this.nextId++,
    mesh: mesh,
    velocity: new THREE.Vector3(),
    isGrabbed: false
  };

  this.objects.push(obj);
  this.scene.add(mesh);

  return obj;
}
toggleSelectionMode(){
  this.selectionMode = !this.selectionMode;

  if(!this.selectionMode){
    // вернуть цвета
    this.objects.forEach(o=>{
      if(o !== this.selectedObject)
        o.mesh.material = this.matNormal;
    });
  }
}
findNearest(palmCenter){

  let nearest = null;
  let minDist = Infinity;

  const handPos = this.palmToWorldOnPlane(palmCenter, 0);
  if(!handPos) return null;

  for(const obj of this.objects){
    const d = handPos.distanceTo(obj.mesh.position);
    if(d < minDist){
      minDist = d;
      nearest = obj;
    }
  }

  return nearest;
}
selectObject(obj){
  if(!obj) return;

  this.selectedObject = obj;

  // reset всех
  this.objects.forEach(o=>{
    o.mesh.material = this.matNormal;
  });

  obj.mesh.material = this.matSelected;

  // pulse animation
  const originalScale = obj.mesh.scale.clone();
  obj.mesh.scale.multiplyScalar(1.2);

  setTimeout(()=>{
    obj.mesh.scale.copy(originalScale);
  }, 200);
}

ndcFromNorm(x,y){
  // зеркалим X правильно
  const nx = (1 - x) * 2 - 1;
  const ny = -(y * 2 - 1);
  return new THREE.Vector2(nx, ny);
}


  palmToWorldOnPlane(palmCenter, planeZ){
    this.plane.set(new THREE.Vector3(0,0,1), -planeZ);
    const ndc = this.ndcFromNorm(palmCenter.x, palmCenter.y);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    const ok = this.raycaster.ray.intersectPlane(this.plane, hit);
    return ok ? hit : null;
  }

updateHover(palmCenter){

  if(this.selectionMode){

    for(const obj of this.objects){
      obj.mesh.material = this.matSelectable;
    }

    const nearest = this.findNearest(palmCenter);
    if(nearest){
      nearest.mesh.material = this.matHover;
    }

    return;
  }

  if(!this.selectedObject) return;

  const mesh = this.selectedObject.mesh;

  if(this.isGrabbed){
    mesh.material = this.matGrab;
    return;
  }

  const handPos = this.palmToWorldOnPlane(palmCenter, mesh.position.z);
  if(!handPos) return;

  const d = handPos.distanceTo(mesh.position);

  mesh.material = d <= this.hoverRadius
    ? this.matHover
    : this.matNormal;
}


tryGrab(palmCenter, pinchDistance){

  if(!this.selectedObject) return false;

  const mesh = this.selectedObject.mesh;

  const handPos = this.palmToWorldOnPlane(palmCenter, mesh.position.z);
  if(!handPos) return false;

  const d = handPos.distanceTo(mesh.position);

  if(d <= this.grabRadius){

    this.isGrabbed = true;

    this.grabOffset.copy(mesh.position).sub(handPos);

    this.basePinchDist = pinchDistance;
    this.baseZ = mesh.position.z;

    mesh.material = this.matGrab;

    return true;
  }

  return false;
}

updateGrab(palmCenter){

  if(!this.isGrabbed || !this.selectedObject) return;

  const mesh = this.selectedObject.mesh;

  const now = performance.now();
  const dt = (now - this.lastFrameTime) / 1000 || 0.016;
  this.lastFrameTime = now;

  const handPos = this.palmToWorldOnPlane(palmCenter, mesh.position.z);
  if(!handPos) return;

  const newPos = handPos.clone().add(this.grabOffset);

  this.selectedObject.velocity
    .copy(newPos)
    .sub(mesh.position)
    .divideScalar(dt);

  mesh.position.copy(newPos);

  this.isInertia = false;
}



updateZoomZ(pinchDistance){

  if(!this.isGrabbed || !this.selectedObject) return;
  if(this.basePinchDist == null) return;

  const mesh = this.selectedObject.mesh;

  const delta = pinchDistance - this.basePinchDist;

  let newZ = this.baseZ + delta * this.zoomGain;

  newZ = clamp(newZ, this.zMin, this.zMax);

  mesh.position.z = newZ;
}


releaseGrab(){

  if(!this.selectedObject) return;

  this.isGrabbed = false;
  this.basePinchDist = null;

  this.isInertia = true;
}

updatePhysics(){

  const now = performance.now();
  const dt = (now - this.lastFrameTime) / 1000 || 0.016;
  this.lastFrameTime = now;

  for(const obj of this.objects){

    if(!obj.velocity) continue;

const move = obj.velocity.clone().multiplyScalar(dt);

// ограничение скорости
if(move.length() > 1.0){
  move.normalize().multiplyScalar(1.0);
}

obj.mesh.position.add(move);


    obj.velocity.multiplyScalar(this.damping);

    if(obj.velocity.length() < 0.01){
      obj.velocity.set(0,0,0);
    }
  }
}

updateTwoHand(palmA, palmB){

  if(!this.selectedObject) return;
  this.selectedObject.velocity.set(0,0,0);


  const mesh = this.selectedObject.mesh;

  const posA = this.palmToWorldOnPlane(palmA, mesh.position.z);
  const posB = this.palmToWorldOnPlane(palmB, mesh.position.z);

  if(!posA || !posB) return;

  const currentDistance = posA.distanceTo(posB);
  const currentAngle = Math.atan2(
    posB.y - posA.y,
    posB.x - posA.x
  );

  if(!this.isDual){
    this.isDual = true;
    this.baseDistance = currentDistance;
    this.baseScale = mesh.scale.x;
    this.baseAngle = currentAngle;
    this.baseRotation = mesh.rotation.z;
    return;
  }

const delta = currentDistance - this.baseDistance;

// deadzone чтобы не дрожало
if(Math.abs(delta) < 0.01) return;

// линейный масштаб (а не относительный)
const scaleSpeed = 2.0;
let newScale = this.baseScale + delta * scaleSpeed;

newScale = clamp(newScale, 0.3, 3.0);

mesh.scale.set(newScale, newScale, newScale);


  const angleDelta = currentAngle - this.baseAngle;
  mesh.rotation.z = this.baseRotation + angleDelta;
}

endTwoHand(){
  this.isDual = false;
}

  render(){
    this.renderer.render(this.scene, this.camera);
  }
}



/* =========================
   Camera + MediaPipe + Loop
========================= */
async function setupCamera(video){
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
}

async function createHandLandmarker(){
  const fileset = await FilesetResolver.forVisionTasks(
    // Важно: tasks-vision грузит wasm/модуль. Vite нормально.
    // Если будут ошибки, можно перейти на CDN версию, но начнём так.
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  return await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
}

async function main(){
  const video = document.getElementById("video");
  const hud = document.getElementById("hud");
  const canvas = document.querySelector(".canvas");

  // 1) camera
  try{
    await setupCamera(video);
  }catch(e){
    hud.textContent = "Camera error: " + e.name + " / " + e.message;
    console.error(e);
    return;
  }

  // 2) three
  const ar = new ARScene(canvas);

  // 3) mediapipe
  hud.textContent = "Loading HandLandmarker…";
  const landmarker = await createHandLandmarker();
  hud.textContent = "Ready";

  // engine parts
  const analyzers = new Map();          // handId -> FingerAnalyzer
  const gestureDetector = new GestureDetector();
  const interaction = new InteractionEngine();

  let lastTimeMs = performance.now();

  function ensureAnalyzer(handId){
    if(!analyzers.has(handId)) analyzers.set(handId, new FingerAnalyzer());
    return analyzers.get(handId);
  }

  function loop(){
    const now = performance.now();
    const res = landmarker.detectForVideo(video, now);

    // IMPORTANT: video зеркальный (scaleX(-1)) → landmarks уже приходят "по изображению"
    // Поэтому отдельно ничего для зеркала делать не надо.

    let status = "no hands";



if(res && res.landmarks && res.landmarks.length){
let palms = [];
let pinchCount = 0;

  let statuses = [];

  // ===== 1. Обрабатываем все руки =====
  for(let i=0;i<res.landmarks.length;i++){

    const handLms = res.landmarks[i];
    const handed = res.handednesses?.[i]?.[0]?.categoryName ?? "Unknown";

    const state = new HandState(handLms, handed);
    palms.push(state.palmCenter);

if(interaction.isPinch(i)){
  pinchCount++;
}

    const analyzer = ensureAnalyzer(i);
    const stable = analyzer.update(handLms, handed);
    const gesture = gestureDetector.detect(stable);
    if(gesture === "PALM"){

if(!ar._spawnTimer){
  ar._spawnTimer = {};
}

if(!ar._spawnTimer[i]){
  ar._spawnTimer[i] = performance.now();
}


if(performance.now() - ar._spawnTimer[i] > 1000){

    const worldPos = ar.palmToWorldOnPlane(state.palmCenter, 0);

    if(worldPos){
      ar.spawnObject(worldPos);
    }

ar._spawnTimer[i] = null;
  }

} else if(ar._spawnTimer && ar._spawnTimer[i]) {
  ar._spawnTimer[i] = null;
}


    const events = interaction.update(i, state, gesture);

    ar.updateHover(state.palmCenter);

for(const ev of events){

  // TOGGLE SELECTION
  if(ev.type==="START" && ev.name==="V_SIGN"){
    ar.toggleSelectionMode();
  }

  // PINCH
  if(ev.type==="START" && ev.name==="PINCH"){

    if(ar.selectionMode){

      const nearest = ar.findNearest(state.palmCenter);
      ar.selectObject(nearest);
      ar.selectionMode = false;

    } else {
      ar.tryGrab(state.palmCenter, state.pinchDistance);
    }
  }

  if(ev.type==="END" && ev.name==="PINCH"){
    ar.releaseGrab();
  }

  if(ev.type==="DRAG"){
    ar.updateGrab(state.palmCenter);
  }
}


    if(interaction.isPinch(i) && ar.isGrabbed){
      ar.updateGrab(state.palmCenter);
      ar.updateZoomZ(state.pinchDistance);
    }


    statuses.push(
      `H${i} ${handed} pinch=${interaction.isPinch(i)}`
    );
  }
// ===== TWO HAND CONTROL =====
if(palms.length === 2 && pinchCount === 2 && ar.selectedObject){

  ar.isGrabbed = false; // отключаем обычный grab
  ar.updateTwoHand(palms[0], palms[1]);

}else{
  ar.endTwoHand();
}

  // ===== 2. Two-hand логика ВНЕ цикла =====
status = `
MODE: ${ar.selectionMode ? "SELECTION" : "CONTROL"}
Objects: ${ar.objects.length}
Selected: ${ar.selectedObject ? ar.selectedObject.id : "none"}
`;

}

if(!res || !res.landmarks || res.landmarks.length < 2){
  ar.endTwoHand();
}

    hud.textContent = status;
ar.updatePhysics();

    // render 3D
    ar.render();

    lastTimeMs = now;
    requestAnimationFrame(loop);
    console.log(res);

  }

  requestAnimationFrame(loop);
}

main();
