
// Globale Variablen für die Three.js-Szene
let scene, camera, renderer, controls;
let raycaster, mouse;
let allObjects = [];
let initialCameraPosition = { x: 0, y: 0, z: 200 };

init();
animate();

function init() {
    // 1. Szene erstellen
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x00000a);

    // 2. Kamera erstellen
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(initialCameraPosition.x, initialCameraPosition.y, initialCameraPosition.z);

    // 3. Renderer erstellen
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('container').appendChild(renderer.domElement);

    // 4. Steuerung (OrbitControls)
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 20;
    controls.maxDistance = 500;

    // 5. Raycaster für Klick-Interaktion
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // 6. Lichter (optional, für mehr Atmosphäre)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
    scene.add(ambientLight);

    // 7. Event Listeners
    window.addEventListener('resize', onWindowResize, false);
    window.addEventListener('mousemove', onMouseMove, false);
    window.addEventListener('click', onMouseClick, false);
    document.getElementById('reset-btn').addEventListener('click', resetView);

    // Daten laden und Szene befüllen
    loadData();
    createStarfield();
}

function loadData() {
    fetch('http://127.0.0.1:5000/api/data')
        .then(response => response.json())
        .then(data => {
            const textureLoader = new THREE.TextureLoader();
            const partyTexture = textureLoader.load('https://threejs.org/examples/textures/sprites/disc.png');

            data.parties.forEach(party => {
                // Partei-Stern erstellen
                const partyMaterial = new THREE.SpriteMaterial({
                    map: partyTexture,
                    color: new THREE.Color(party.color),
                    transparent: true,
                    blending: THREE.AdditiveBlending,
                    depthWrite: false,
                    opacity: 0.8
                });
                const partySprite = new THREE.Sprite(partyMaterial);
                partySprite.position.set(party.x, party.y, party.z);
                partySprite.scale.set(15, 15, 1); // Parteien sind größer
                partySprite.userData = { type: 'party', ...party };
                scene.add(partySprite);
                allObjects.push(partySprite);

                // Zugehörige Politiker-Planeten erstellen
                party.politicians.forEach(politician => {
                    const politicianMaterial = new THREE.SpriteMaterial({
                        map: partyTexture,
                        color: new THREE.Color(party.color),
                        transparent: true,
                        opacity: 0.7
                    });
                    const politicianSprite = new THREE.Sprite(politicianMaterial);
                    politicianSprite.position.set(politician.x, politician.y, politician.z);
                    politicianSprite.scale.set(5, 5, 1); // Politiker sind kleiner
                    politicianSprite.userData = { type: 'politician', ...politician };
                    scene.add(politicianSprite);
                    allObjects.push(politicianSprite);
                });
            });
        });
}

function createStarfield() {
    const starVertices = [];
    for (let i = 0; i < 10000; i++) {
        const x = THREE.MathUtils.randFloatSpread(2000);
        const y = THREE.MathUtils.randFloatSpread(2000);
        const z = THREE.MathUtils.randFloatSpread(2000);
        starVertices.push(x, y, z);
    }
    const starGeometry = new THREE.BufferGeometry();
    starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
    const starMaterial = new THREE.PointsMaterial({ color: 0x888888, size: 0.7 });
    const starfield = new THREE.Points(starGeometry, starMaterial);
    scene.add(starfield);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function onMouseMove(event) {
    // Mauskoordinaten für Raycaster normalisieren
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    // Tooltip anzeigen
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(allObjects);
    const tooltipElement = document.getElementById('tooltip');
    
    if (intersects.length > 0) {
        const d = intersects[0].object.userData;
        tooltipElement.style.visibility = 'visible';
        tooltipElement.style.left = event.clientX + 15 + 'px';
        tooltipElement.style.top = event.clientY + 'px';
        tooltipElement.innerHTML = d.type === 'party' 
            ? `<b>${d.name}</b><br><i>Sternensystem</i>` 
            : `<b>${d.name}</b><br>Schwerpunkte: ${d.prominent_words.join(', ')}`;
    } else {
        tooltipElement.style.visibility = 'hidden';
    }
}

function onMouseClick() {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(allObjects);

    if (intersects.length > 0 && intersects[0].object.userData.type === 'party') {
        const clickedParty = intersects[0].object;
        focusOnParty(clickedParty);
    }
}

function focusOnParty(partyObject) {
    const partyData = partyObject.userData;
    
    // Position, auf die die Kamera schauen soll (der Stern der Partei)
    const targetPosition = partyObject.position;

    // Kamera sanft zur Partei bewegen
    new TWEEN.Tween(camera.position)
        .to({ x: targetPosition.x, y: targetPosition.y, z: targetPosition.z + 50 }, 1500)
        .easing(TWEEN.Easing.Quadratic.InOut)
        .start();

    // Steuerung anweisen, auf das neue Ziel zu blicken
    new TWEEN.Tween(controls.target)
        .to(targetPosition, 1500)
        .easing(TWEEN.Easing.Quadratic.InOut)
        .start();
        
    // Andere Objekte ausblenden, das gewählte System hervorheben
    allObjects.forEach(obj => {
        const isRelated = obj.userData.id === partyData.id || obj.userData.partyId === partyData.id;
        new TWEEN.Tween(obj.material)
            .to({ opacity: isRelated ? 1 : 0.05 }, 1000)
            .start();
    });

    // Reset-Button anzeigen
    document.getElementById('reset-btn').classList.remove('hidden');
}

function resetView() {
     // Kamera zurück zur Startposition bewegen
     new TWEEN.Tween(camera.position)
        .to(initialCameraPosition, 1500)
        .easing(TWEEN.Easing.Quadratic.InOut)
        .start();

    // Steuerung anweisen, wieder auf den Ursprung zu blicken
    new TWEEN.Tween(controls.target)
        .to({ x: 0, y: 0, z: 0 }, 1500)
        .easing(TWEEN.Easing.Quadratic.InOut)
        .start();

    // Alle Objekte wieder einblenden
    allObjects.forEach(obj => {
        const originalOpacity = obj.userData.type === 'party' ? 0.8 : 0.7;
        new TWEEN.Tween(obj.material)
            .to({ opacity: originalOpacity }, 1000)
            .start();
    });
    
    // Reset-Button verstecken
    document.getElementById('reset-btn').classList.add('hidden');
}


function animate() {
    requestAnimationFrame(animate);
    controls.update(); // Wichtig für Damping
    TWEEN.update(); // Wichtig für Kamera-Animationen
    renderer.render(scene, camera);
}