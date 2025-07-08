/**
 * Political Latent Space - 3D Galaxy Visualization
 * 
 * This file implements a 3D galaxy visualization using D3.js for data binding
 * and Three.js for WebGL rendering, inspired by the test implementation but
 * integrated with the main project's data structure.
 */

// Main function to create the 3D galaxy visualization
function createGalaxyVisualization(data, containerId) {
    // Container for the visualization
    const container = document.getElementById(containerId);
    
    // Set up Three.js scene
    let scene, camera, renderer, controls;
    let raycaster, mouse;
    let allObjects = [];
    
    // Initialize the scene
    initScene();
    
    // Process and visualize data
    visualizeData(data);
    
    // Start animation loop
    animate();
    
    // Function to initialize the Three.js scene
    function initScene() {
        // Create scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x00000a);
        
        // Create camera
        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(0, 0, 200);
        
        // Create renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);
        
        // Add orbit controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = false;
        controls.minDistance = 20;
        controls.maxDistance = 500;
        
        // Set up raycaster for interactions
        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();
        
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
        scene.add(ambientLight);
        
        // Add event listeners
        window.addEventListener('resize', onWindowResize, false);
        container.addEventListener('mousemove', onMouseMove, false);
        container.addEventListener('click', onMouseClick, false);
        
        // Create starfield background
        createStarfield();
    }
    
    // Function to visualize the data
    function visualizeData(data) {
        // Extract entities from data
        const movements = data.movements || [];
        const politicians = data.politicians || [];
        
        // Load textures
        const textureLoader = new THREE.TextureLoader();
        const orbTexture = textureLoader.load('/static/assets/star.png');
        
        // Create a D3 scale for size based on influence
        const sizeScale = d3.scaleLinear()
            .domain([0, d3.max([...movements, ...politicians], d => d.influence || 1)])
            .range([10, 25]);
            
        // Create movements (orbs)
        movements.forEach(movement => {
            // Use D3 to calculate position based on political dimensions
            const x = movement.position ? movement.position.economic_axis * 200 - 100 : (Math.random() * 200 - 100);
            const y = movement.position ? movement.position.social_axis * 200 - 100 : (Math.random() * 200 - 100);
            const z = movement.position ? movement.position.ecological_axis * 200 - 100 : (Math.random() * 200 - 100);
            
            // Create orb material with movement color
            const color = movement.color || getColorFromPosition(movement.position);
            const orbMaterial = new THREE.SpriteMaterial({
                map: orbTexture,
                color: new THREE.Color(color),
                transparent: true,
                blending: THREE.AdditiveBlending,
                depthWrite: false,
                opacity: 0.8
            });
            
            // Create movement orb sprite
            const movementSprite = new THREE.Sprite(orbMaterial);
            movementSprite.position.set(x, y, z);
            movementSprite.scale.set(sizeScale(movement.influence || 1), sizeScale(movement.influence || 1), 1);
            movementSprite.userData = { type: 'movement', ...movement };
            scene.add(movementSprite);
            allObjects.push(movementSprite);
            
            // Add label for the movement
            addLabel(movement.name, x, y + sizeScale(movement.influence || 1) + 5, z);
            
            // Create politicians (planets) for this movement
            const movementPoliticians = politicians.filter(p => p.movement === movement.id);
            movementPoliticians.forEach(politician => {
                // Calculate position relative to the movement
                const distance = 20 + Math.random() * 15;
                const angle = Math.random() * Math.PI * 2;
                const height = (Math.random() - 0.5) * 10;
                
                const px = x + Math.cos(angle) * distance;
                const py = y + height;
                const pz = z + Math.sin(angle) * distance;
                
                // Create politician material - use the same texture but with different color
                const politicianColor = politician.position ? getColorFromPosition(politician.position) : color;
                const politicianMaterial = new THREE.SpriteMaterial({
                    map: orbTexture,
                    color: new THREE.Color(politicianColor),
                    transparent: true,
                    opacity: 0.7
                });
                
                // Create politician sprite
                const politicianSprite = new THREE.Sprite(politicianMaterial);
                politicianSprite.position.set(px, py, pz);
                politicianSprite.scale.set(sizeScale(politician.influence || 0.5) * 0.4, sizeScale(politician.influence || 0.5) * 0.4, 1);
                politicianSprite.userData = { type: 'politician', ...politician, movementId: movement.id };
                scene.add(politicianSprite);
                allObjects.push(politicianSprite);
            });
        });
    }
    
    // Function to create starfield background
    function createStarfield() {
        // Use D3 to generate random star positions
        const starCount = 10000;
        const starPositions = d3.range(starCount).map(() => ({
            x: THREE.MathUtils.randFloatSpread(2000),
            y: THREE.MathUtils.randFloatSpread(2000),
            z: THREE.MathUtils.randFloatSpread(2000)
        }));
        
        // Create geometry from positions
        const starVertices = [];
        starPositions.forEach(pos => {
            starVertices.push(pos.x, pos.y, pos.z);
        });
        
        const starGeometry = new THREE.BufferGeometry();
        starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
        
        // Create material and points
        const starMaterial = new THREE.PointsMaterial({ color: 0x888888, size: 0.7 });
        const starfield = new THREE.Points(starGeometry, starMaterial);
        
        // Add to scene
        scene.add(starfield);
    }
    
    // Function to add text label (optional)
    function addLabel(text, x, y, z) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;
        
        context.font = '24px Arial';
        context.fillStyle = 'white';
        context.textAlign = 'center';
        context.fillText(text, 128, 32);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
        const sprite = new THREE.Sprite(material);
        
        sprite.position.set(x, y, z);
        sprite.scale.set(20, 5, 1);
        
        scene.add(sprite);
    }
    
    // Function to get color from position
    function getColorFromPosition(position) {
        if (!position) return '#aaaaaa';
        
        // Use D3 color scales based on political dimensions
        const economicScale = d3.scaleLinear()
            .domain([0, 0.5, 1])
            .range(['#e63946', '#aaaaaa', '#457b9d']);
            
        const socialScale = d3.scaleLinear()
            .domain([0, 0.5, 1])
            .range(['#6a0dad', '#aaaaaa', '#ffb703']);
            
        const ecologicalScale = d3.scaleLinear()
            .domain([0, 0.5, 1])
            .range(['#d62828', '#aaaaaa', '#2a9d8f']);
        
        // Mix colors based on position
        const economic = position.economic_axis || 0.5;
        const social = position.social_axis || 0.5;
        const ecological = position.ecological_axis || 0.5;
        
        // Use D3 color interpolation
        return d3.interpolateRgb(
            d3.interpolateRgb(economicScale(economic), socialScale(social))(0.5),
            ecologicalScale(ecological)
        )(0.5);
    }
    
    // Event handler for window resize
    function onWindowResize() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }
    
    // Event handler for mouse movement
    function onMouseMove(event) {
        // Calculate mouse position in normalized device coordinates
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Update tooltip
        updateTooltip();
    }
    
    // Function to update tooltip
    function updateTooltip() {
        // Cast ray from mouse position
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(allObjects);
        
        // Get tooltip element
        const tooltip = document.getElementById('galaxy-tooltip') || createTooltip();
        
        if (intersects.length > 0) {
            const object = intersects[0].object;
            const data = object.userData;
            
            // Position tooltip near mouse
            const rect = renderer.domElement.getBoundingClientRect();
            tooltip.style.left = (event.clientX + 15) + 'px';
            tooltip.style.top = event.clientY + 'px';
            
            // Set tooltip content
            tooltip.innerHTML = `
                <strong>${data.name}</strong><br>
                <em>${data.type.charAt(0).toUpperCase() + data.type.slice(1)}</em>
                ${data.position ? `<br>
                Economic: ${(data.position.economic_axis * 2 - 1).toFixed(2)}<br>
                Social: ${(data.position.social_axis * 2 - 1).toFixed(2)}<br>
                Ecological: ${(data.position.ecological_axis * 2 - 1).toFixed(2)}` : ''}
            `;
            
            // Show tooltip
            tooltip.style.display = 'block';
        } else {
            // Hide tooltip
            tooltip.style.display = 'none';
        }
    }
    
    // Function to create tooltip
    function createTooltip() {
        // Use the tooltip element already defined in the HTML
        const tooltip = document.getElementById('galaxy-tooltip');
        if (!tooltip) {
            // If not found, create it
            const newTooltip = document.createElement('div');
            newTooltip.id = 'galaxy-tooltip';
            newTooltip.style.position = 'absolute';
            newTooltip.style.padding = '10px';
            newTooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
            newTooltip.style.color = '#fff';
            newTooltip.style.borderRadius = '5px';
            newTooltip.style.pointerEvents = 'none';
            newTooltip.style.zIndex = '1000';
            newTooltip.style.display = 'none';
            newTooltip.style.fontSize = '14px';
            newTooltip.style.maxWidth = '200px';
            newTooltip.style.border = '1px solid rgba(255, 255, 255, 0.2)';
            document.body.appendChild(newTooltip);
            return newTooltip;
        }
        return tooltip;
    }
    
    // Event handler for mouse click
    function onMouseClick(event) {
        // Update mouse position
        mouse.x = (event.clientX / container.clientWidth) * 2 - 1;
        mouse.y = -(event.clientY / container.clientHeight) * 2 + 1;
        
        // Check for intersections
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(allObjects);
        
        if (intersects.length > 0) {
            const clickedObject = intersects[0].object;
            const data = clickedObject.userData;
            
            if (data.type === 'movement') {
                // Focus on the movement
                focusOnEntity('movement', data.id);
                
                // Highlight this movement and its politicians, dim others
                allObjects.forEach(obj => {
                    const isRelated = obj.userData.id === data.id || obj.userData.movementId === data.id;
                    // Use D3 for smooth transition
                    d3.select(obj.material)
                      .transition()
                      .duration(1000)
                      .tween('opacity', () => {
                          const startOpacity = obj.material.opacity;
                          const targetOpacity = isRelated ? (obj.userData.type === 'movement' ? 0.8 : 0.7) : 0.05;
                          return t => { obj.material.opacity = startOpacity + (targetOpacity - startOpacity) * t; };
                      });
                });
            } else {
                // For politicians, trigger the focus view
                loadEntityFocusView(data.type, data.id);
            }
        }
    }
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        updateTooltip(); // Update tooltip on each frame
        renderer.render(scene, camera);
    }
    
    // Return public methods
    return {
        // Method to focus on a specific entity
        focusOnEntity: function(entityType, entityName) {
            const entity = allObjects.find(obj => 
                obj.userData.type === entityType && 
                obj.userData.name === entityName
            );
            
            if (entity) {
                // Animate camera to focus on entity
                const targetPosition = entity.position.clone();
                const startPosition = camera.position.clone();
                
                // Use D3 for smooth transition
                d3.transition()
                    .duration(1500)
                    .tween('camera', () => {
                        return (t) => {
                            // Interpolate camera position
                            camera.position.x = startPosition.x + (targetPosition.x + 50 - startPosition.x) * t;
                            camera.position.y = startPosition.y + (targetPosition.y - startPosition.y) * t;
                            camera.position.z = startPosition.z + (targetPosition.z + 50 - startPosition.z) * t;
                            
                            // Look at target
                            camera.lookAt(targetPosition);
                            camera.updateProjectionMatrix();
                        };
                    });
                
                // Update controls target
                controls.target.copy(targetPosition);
            }
        },
        
        // Method to reset view
        resetView: function() {
            // Animate camera back to initial position
            const startPosition = camera.position.clone();
            const targetPosition = new THREE.Vector3(0, 0, 200);
            
            // Use D3 for smooth transition
            d3.transition()
                .duration(1500)
                .tween('camera', () => {
                    return (t) => {
                        // Interpolate camera position
                        camera.position.x = startPosition.x + (targetPosition.x - startPosition.x) * t;
                        camera.position.y = startPosition.y + (targetPosition.y - startPosition.y) * t;
                        camera.position.z = startPosition.z + (targetPosition.z - startPosition.z) * t;
                        
                        // Look at center
                        camera.lookAt(new THREE.Vector3(0, 0, 0));
                        camera.updateProjectionMatrix();
                    };
                });
            
            // Reset controls target
            controls.target.set(0, 0, 0);
        }
    };
}

// Function to load the galaxy visualization
function loadGalaxyVisualization() {
    // Show loading indicator
    document.getElementById('galaxy-container').innerHTML = '<div class="loading">Loading galaxy visualization...</div>';
    
    // Fetch the galaxy data
    fetch('/api/galaxy-data')
        .then(response => response.json())
        .then(data => {
            // Clear loading indicator
            document.getElementById('galaxy-container').innerHTML = '';
            
            // Create the visualization
            window.galaxyVis = createGalaxyVisualization(data, 'galaxy-container');
            
            // Store reference to visualization for later use
            window.galaxyVis = galaxyVis;
        })
        .catch(error => {
            console.error('Error loading galaxy data:', error);
            document.getElementById('galaxy-container').innerHTML = 
                '<div class="error">Error loading visualization: ' + error.message + '</div>';
        });
}